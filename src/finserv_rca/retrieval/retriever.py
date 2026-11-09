"""Hybrid retriever: dense + BM25 -> RRF -> re-rank -> contextual compression."""
from __future__ import annotations

import re
from datetime import date

from ..config import Settings
from ..ingest.chunker import split_sentences
from ..models import Chunk, RetrievedChunk
from .embeddings import get_embedder
from .rerank import build_reranker
from .sparse import SparseIndex, tokenize
from .vector_store import VectorStore


class HybridRetriever:
    def __init__(self, settings: Settings, store: VectorStore) -> None:
        self.cfg = settings.retrieval
        self.embedder = get_embedder(settings.embedding.backend, settings.embedding.model)
        self.store = store
        self.reranker = build_reranker(
            self.cfg.reranker, self.embedder, self.cfg.cross_encoder_model
        )
        self.sparse = SparseIndex(store.all_chunks())

    def refresh(self) -> None:
        self.sparse = SparseIndex(self.store.all_chunks())

    def expand(self, query: str) -> str:
        """Add glossary expansions, so 'CDD' also matches 'customer due diligence'."""
        extras = [
            expansion
            for term, expansion in self.cfg.glossary.items()
            if re.search(rf"\b{re.escape(term)}\b", query, re.IGNORECASE)
        ]
        return f"{query} {' '.join(extras)}" if extras else query

    def retrieve(
        self,
        query: str,
        as_of: date | None = None,
        frameworks: list[str] | None = None,
        k: int | None = None,
    ) -> list[RetrievedChunk]:
        expanded = self.expand(query)
        dense: list[tuple[Chunk, float]] = []
        sparse: list[tuple[Chunk, float]] = []
        if self.cfg.mode in ("dense", "hybrid"):
            vector = self.embedder.embed([expanded])[0]
            dense = self.store.search(vector, self.cfg.dense_k, as_of, frameworks)
        if self.cfg.mode in ("sparse", "hybrid"):
            sparse = self.sparse.search(expanded, self.cfg.sparse_k, as_of, frameworks)
        fused = self._fuse(dense, sparse)[: self.cfg.rerank_top_n]
        # Glossary terms widen recall. Scoring on them would reward every KYC chunk.
        ranked = self.reranker.rerank(query, fused)
        top = ranked[: k or self.cfg.final_k]
        for item in top:
            item.compressed_text = self._compress(expanded, item.chunk.text)
        return top

    def _fuse(self, dense, sparse) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion: combines ranks, so the two score scales need no calibration."""
        merged: dict[str, RetrievedChunk] = {}
        for attr, results in (("dense_rank", dense), ("sparse_rank", sparse)):
            for rank, (chunk, _) in enumerate(results, start=1):
                item = merged.setdefault(chunk.chunk_id, RetrievedChunk(chunk=chunk, score=0.0))
                item.score += 1.0 / (self.cfg.rrf_k + rank)
                setattr(item, attr, rank)
        return sorted(merged.values(), key=lambda i: i.score, reverse=True)

    def _compress(self, query: str, text: str) -> str:
        """Keep the most query-relevant sentences in original order. Never rewrites legal text."""
        sentences = split_sentences(text)
        limit = self.cfg.compression_max_sentences
        if len(sentences) <= limit:
            return text
        terms = set(tokenize(query))
        scored = sorted(
            range(len(sentences)),
            key=lambda i: len(terms & set(tokenize(sentences[i]))),
            reverse=True,
        )
        keep = sorted(scored[:limit])
        return " ".join(sentences[i] for i in keep)
