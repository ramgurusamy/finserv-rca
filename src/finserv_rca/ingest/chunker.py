"""Structure-first chunking.

Approach: keep one numbered paragraph as one chunk whenever it fits, because a
paragraph is the citable legal unit. Only oversized paragraphs are windowed, on
sentence boundaries with overlap, and every window keeps the same para_ref.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date

from ..models import Chunk
from .parsers import Section

SENTENCE_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def count_tokens(text: str) -> int:
    """Word count is a deliberate, dependency-free proxy for model tokens."""
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


def window_section(text: str, max_tokens: int, overlap: int) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in split_sentences(text):
        length = count_tokens(sentence)
        if current and size + length > max_tokens:
            windows.append(" ".join(current))
            kept: list[str] = []
            kept_size = 0
            for previous in reversed(current):
                if kept_size + count_tokens(previous) > overlap:
                    break
                kept.insert(0, previous)
                kept_size += count_tokens(previous)
            current, size = kept, kept_size
        current.append(sentence)
        size += length
    if current:
        windows.append(" ".join(current))
    return windows


def make_chunks(meta: dict, sections: list[Section], max_tokens: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        for ordinal, text in enumerate(window_section(section.text, max_tokens, overlap)):
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            chunk_key = f"{meta['doc_id']}|{meta['version']}|{section.para_ref}|{ordinal}"
            chunks.append(
                Chunk(
                    chunk_id=hashlib.sha256(chunk_key.encode()).hexdigest()[:32],
                    doc_id=meta["doc_id"],
                    title=meta["title"],
                    framework=meta["framework"],
                    jurisdiction=meta["jurisdiction"],
                    data_residency=meta.get("data_residency", meta["jurisdiction"]),
                    version=str(meta["version"]),
                    effective_from=date.fromisoformat(str(meta["effective_from"])),
                    para_ref=section.para_ref,
                    heading=section.heading,
                    text=text,
                    source_url=meta["source_url"],
                    reference=meta.get("reference", ""),
                    content_class=meta.get("content_class", "digest"),
                    content_hash=content_hash,
                )
            )
    return chunks
