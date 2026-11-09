"""Custom evaluation metrics with the same names and intent as RAGAS.

Approach: retrieval metrics use labelled paragraph references, so they are exact
and need no model. Generation metrics are deterministic proxies. An optional LLM
judge adds a RAGAS-style faithfulness check on the laptop profile.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from ..guardrails.output import LABEL_RE, numbers_in, split_answer
from ..models import RetrievedChunk
from ..retrieval.sparse import tokenize

SUPPORT_THRESHOLD = 0.6


def ref_matches(gold: str, retrieved: str) -> bool:
    """Official parsers split at paragraph level, so 'KYC-Para 17' covers 'KYC-Para 17(3)'."""
    return gold == retrieved or gold.startswith(f"{retrieved}(")


def context_recall(retrieved_refs: list[str], gold_refs: list[str]) -> float:
    """Share of gold paragraphs found anywhere in the retrieved set."""
    if not gold_refs:
        return 1.0
    found = sum(any(ref_matches(g, r) for r in retrieved_refs) for g in set(gold_refs))
    return found / len(set(gold_refs))


def context_precision(retrieved_refs: list[str], gold_refs: list[str]) -> float:
    """Average precision at each relevant rank, the same form RAGAS uses."""
    gold = set(gold_refs)
    if not gold:
        return 1.0
    hits = 0
    total = 0.0
    unmatched = set(gold)
    for rank, ref in enumerate(retrieved_refs, start=1):
        matched = {g for g in unmatched if ref_matches(g, ref)}
        if matched:
            unmatched -= matched
            hits += 1
            total += hits / rank
    return total / len(gold)


def split_claims(answer: str) -> list[str]:
    return [s for s in split_answer(answer) if LABEL_RE.sub("", s).strip()]


def claim_supported(claim: str, contexts: Iterable[RetrievedChunk]) -> bool:
    """A claim is supported if a single context covers its terms and all its numbers."""
    plain = LABEL_RE.sub(" ", claim)
    terms = set(tokenize(plain))
    numbers = numbers_in(plain)
    if not terms:
        return True
    for item in contexts:
        words = set(tokenize(item.chunk.text))
        coverage = len(terms & words) / len(terms)
        if coverage >= SUPPORT_THRESHOLD and numbers <= numbers_in(item.chunk.text):
            return True
    return False


def faithfulness(answer: str, contexts: list[RetrievedChunk]) -> float:
    claims = split_claims(answer)
    if not claims:
        return 0.0
    return sum(claim_supported(c, contexts) for c in claims) / len(claims)


def answer_relevance(question: str, answer: str, embedder) -> float:
    """Cosine similarity of question and answer, plus question term coverage."""
    plain = LABEL_RE.sub(" ", answer)
    vectors = embedder.embed([question, plain])
    cosine = float(np.clip(vectors[0] @ vectors[1], 0.0, 1.0))
    terms = set(tokenize(question))
    coverage = len(terms & set(tokenize(plain))) / max(len(terms), 1)
    return 0.5 * cosine + 0.5 * coverage


def fact_coverage(answer: str, facts: list[str]) -> float:
    """Ground-truth check: required facts must appear in the answer."""
    if not facts:
        return 1.0
    plain = re.sub(r"\s+", " ", answer.lower())
    return sum(fact.lower() in plain for fact in facts) / len(facts)
