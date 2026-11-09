"""Re-ranking backends.

`feature` is an offline, explainable scorer used when model downloads are not
possible. `cross_encoder` is the intended quality path on the laptop profile.
Both return scores in [0, 1], so the abstain threshold works the same way.
"""
from __future__ import annotations

import math
import re
from itertools import pairwise
from typing import Protocol

import numpy as np

from ..models import RetrievedChunk
from .sparse import tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class NoReranker:
    name = "none"

    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        top = max((item.score for item in items), default=1.0) or 1.0
        for item in items:
            item.score = item.score / top
        return items


class FeatureReranker:
    """Weighted blend of fusion rank, term coverage, phrase hits, heading match and similarity.

    The fusion score stays in the blend as a prior. An earlier version dropped it, and
    the ablation study showed that this made ranking worse than RRF alone.
    """

    name = "feature-v1"
    REF_RE = re.compile(r"\b(?:[A-Z]{2,5}\d{2}\.\d+|Art(?:icle)?\s*\d+|Para(?:graph)?\s*\d+)", re.IGNORECASE)

    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not items:
            return items
        terms = list(dict.fromkeys(tokenize(query)))
        documents = [tokenize(f"{i.chunk.heading} {i.chunk.text}") for i in items]
        idf = {
            term: math.log(1 + len(items) / (1 + sum(term in d for d in documents)))
            for term in terms
        }
        total_idf = sum(idf.values()) or 1.0
        bigrams = list(pairwise(terms))
        vectors = self.embedder.embed([query] + [i.chunk.text for i in items])
        cosine = vectors[1:] @ vectors[0]
        refs = {r.lower().replace(" ", "") for r in self.REF_RE.findall(query)}
        top_fusion = max(item.score for item in items) or 1.0
        for item, words, similarity in zip(items, documents, cosine):
            prior = item.score / top_fusion
            word_set = set(words)
            coverage = sum(idf[t] for t in terms if t in word_set) / total_idf
            joined = " ".join(words)
            phrase = (
                sum(f"{a} {b}" in joined for a, b in bigrams) / len(bigrams) if bigrams else 0.0
            )
            heading_words = set(tokenize(f"{item.chunk.para_ref} {item.chunk.heading}"))
            heading = len(heading_words & set(terms)) / max(len(set(terms)), 1)
            para = item.chunk.para_ref.lower().replace(" ", "")
            exact = 1.0 if any(ref in para or para in ref for ref in refs) else 0.0
            similarity = max(float(similarity), 0.0)
            # Ranking keeps the fusion prior. Abstention must not, because the top fused
            # result always has a prior of 1, even for an off-topic question.
            item.evidence = float(0.6 * coverage + 0.2 * phrase + 0.2 * similarity)
            item.score = float(
                0.30 * prior
                + 0.35 * coverage
                + 0.10 * phrase
                + 0.10 * min(heading * 2, 1.0)
                + 0.10 * similarity
                + 0.05 * exact
            )
        return sorted(items, key=lambda i: i.score, reverse=True)


class CrossEncoderReranker:
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model)
        self.name = model

    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not items:
            return items
        logits = np.asarray(
            self._model.predict([(query, item.chunk.text) for item in items], show_progress_bar=False)
        )
        for item, logit in zip(items, logits):
            item.score = float(1 / (1 + math.exp(-float(logit))))
        return sorted(items, key=lambda i: i.score, reverse=True)


def build_reranker(kind: str, embedder, model: str) -> Reranker:
    if kind == "none":
        return NoReranker()
    if kind == "feature":
        return FeatureReranker(embedder)
    if kind == "cross_encoder":
        return CrossEncoderReranker(model)
    raise ValueError(f"Unknown reranker: {kind}")
