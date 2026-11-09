"""BM25 keyword index. Regulatory search depends on exact tokens such as RBC20.1 or 4.5%."""
from __future__ import annotations

import re
from datetime import date

from rank_bm25 import BM25Okapi

from ..models import Chunk

TOKEN_RE = re.compile(r"[a-z]+\d+(?:\.\d+)?|\d+(?:\.\d+)?%?|[a-z][a-z\-']+")
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have", "how", "if", "in", "is", "it", "its", "may", "must", "of", "on", "or", "that", "the", "their", "there", "these", "this", "to", "under", "what", "when", "where", "which", "who", "will", "with", "would", "should", "than", "then", "they"]
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class SparseIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        corpus = [
            tokenize(f"{c.para_ref} {c.heading} {c.title} {c.text}") for c in chunks
        ]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(
        self,
        query: str,
        limit: int,
        as_of: date | None = None,
        frameworks: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        day = as_of or date.today()
        scores = self._bm25.get_scores(tokenize(query))
        hits = []
        for chunk, score in zip(self.chunks, scores):
            if score <= 0 or chunk.effective_from > day:
                continue
            if chunk.effective_to and chunk.effective_to <= day:
                continue
            if frameworks and chunk.framework not in frameworks:
                continue
            hits.append((chunk, float(score)))
        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[:limit]
