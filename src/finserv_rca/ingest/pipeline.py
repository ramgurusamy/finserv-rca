"""Idempotent, versioned ingestion.

Content hashes make re-runs no-ops. When a new version of a paragraph arrives the
older chunk is closed with `effective_to` and kept. Partial amendments therefore
only replace the paragraphs they touch.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import Settings
from ..models import Chunk
from ..retrieval.embeddings import get_embedder
from ..retrieval.vector_store import VectorStore
from .chunker import make_chunks
from .parsers import (
    ParseError,
    read_html_text,
    read_markdown,
    read_pdf_text,
    split_by_paragraph_numbers,
)

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    documents: int = 0
    added: int = 0
    unchanged: int = 0
    superseded: int = 0
    failed: list[str] = field(default_factory=list)
    changed_refs: list[dict] = field(default_factory=list)


class Ingestor:
    def __init__(self, settings: Settings, store: VectorStore) -> None:
        self.settings = settings
        self.store = store
        self.embedder = get_embedder(settings.embedding.backend, settings.embedding.model)

    def _official_entry(self, entry: dict) -> tuple[dict, list]:
        path = self.settings.path(self.settings.ingest.official_dir) / entry["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        text = read_pdf_text(path) if path.suffix == ".pdf" else read_html_text(path)
        sections = []
        for rule in entry.get("para_rules", []):
            sections = split_by_paragraph_numbers(text, rule["pattern"], rule.get("prefix", ""))
            if len(sections) > 3:
                break
        if not sections:
            raise ParseError(f"{path.name}: no paragraph rules matched")
        return {**entry["meta"], "content_class": "official"}, sections

    def load_corpus(self) -> list[tuple[str, dict, list]]:
        """Digest corpus by default. Official files replace digests when they are present."""
        cfg = self.settings.ingest
        documents = {}
        for path in sorted(self.settings.path(cfg.corpus_dir).glob("*.md")):
            meta, sections = read_markdown(path)
            documents[meta["doc_id"]] = (path.name, meta, sections)
        if cfg.use_official:
            manifest = yaml.safe_load(self.settings.path(cfg.sources_manifest).read_text())
            for entry in manifest["official"]:
                try:
                    meta, sections = self._official_entry(entry)
                    documents[meta["doc_id"]] = (entry["file"], meta, sections)
                except (FileNotFoundError, ParseError) as exc:
                    log.warning("Keeping digest for %s: %s", entry["meta"]["doc_id"], exc)
        return list(documents.values())

    def ingest_all(self) -> IngestResult:
        result = IngestResult()
        for name, meta, sections in self.load_corpus():
            self._ingest_document(name, meta, sections, result)
        self._write_registry()
        return result

    def ingest_file(self, path: Path) -> IngestResult:
        """Ingest one new circular, for example a markdown digest or an amendment."""
        result = IngestResult()
        try:
            meta, sections = read_markdown(path)
            self._ingest_document(path.name, meta, sections, result)
        except (ParseError, OSError) as exc:
            result.failed.append(f"{path.name}: {exc}")
        self._write_registry()
        return result

    def _ingest_document(self, name, meta, sections, result: IngestResult) -> None:
        cfg = self.settings.ingest
        try:
            chunks = make_chunks(meta, sections, cfg.max_chunk_tokens, cfg.overlap_tokens)
        except (KeyError, ValueError) as exc:
            result.failed.append(f"{name}: invalid metadata ({exc})")
            return
        existing = [c for c in self.store.all_chunks() if c.doc_id == meta["doc_id"]]
        known = {c.content_hash for c in existing if c.version == str(meta["version"])}
        new = [c for c in chunks if c.content_hash not in known]
        result.documents += 1
        result.unchanged += len(chunks) - len(new)
        if not new:
            return
        close_on = new[0].effective_from
        new_refs = {c.para_ref for c in new}
        for old in existing:
            if (
                old.para_ref in new_refs
                and old.version != new[0].version
                and old.effective_to is None
                and old.effective_from < close_on
            ):
                self.store.close_version(old, close_on)
                result.superseded += 1
                result.changed_refs.append(
                    {
                        "doc_id": old.doc_id,
                        "para_ref": old.para_ref,
                        "old_version": old.version,
                        "new_version": new[0].version,
                        "old_text": old.text,
                    }
                )
        for chunk in new:
            if chunk.para_ref not in {c.para_ref for c in existing}:
                result.changed_refs.append(
                    {
                        "doc_id": chunk.doc_id,
                        "para_ref": chunk.para_ref,
                        "old_version": None,
                        "new_version": chunk.version,
                        "old_text": "",
                    }
                )
        vectors = self.embedder.embed([self._embedding_text(c) for c in new])
        self.store.upsert(new, vectors)
        result.added += len(new)

    @staticmethod
    def _embedding_text(chunk: Chunk) -> str:
        """Contextual prefix: an isolated clause should still know its regulation."""
        return f"{chunk.framework} {chunk.para_ref} {chunk.heading}. {chunk.text}"

    def _write_registry(self) -> None:
        """Document registry: which version of what is live, with provenance."""
        registry: dict[str, dict] = {}
        for chunk in self.store.all_chunks():
            entry = registry.setdefault(
                f"{chunk.doc_id}@{chunk.version}",
                {
                    "doc_id": chunk.doc_id,
                    "version": chunk.version,
                    "title": chunk.title,
                    "reference": chunk.reference,
                    "source_url": chunk.source_url,
                    "content_class": chunk.content_class,
                    "effective_from": chunk.effective_from.isoformat(),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        path = self.settings.path(self.settings.paths.registry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(registry.values(), key=lambda e: e["doc_id"]), indent=2))


