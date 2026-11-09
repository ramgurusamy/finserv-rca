"""Deterministic output validation. Nothing reaches a user without passing this."""
from __future__ import annotations

import re

from ..models import GuardrailReport, RetrievedChunk

LABEL_RE = re.compile(r"\[S(\d+)\]")
NUMBER_RE = re.compile(r"(?<![A-Za-z\d.])\d+(?:[.,]\d+)*(?![A-Za-z\d])%?")
ANSWER_SPLIT_RE = re.compile(r"(?<=\])\.?\s+(?=[A-Z(])|(?<=[.;:])\s+(?=[A-Z(])")
INJECTION_RE = re.compile(
    r"(?i)ignore (?:all |any )?(?:previous|prior|above) (?:instructions|rules)"
    r"|disregard (?:the )?system|reveal (?:the |your )?(?:system )?prompt"
)


def split_answer(answer: str) -> list[str]:
    """Split generated text into claims. A claim's citation may follow its full stop."""
    return [part.strip() for part in ANSWER_SPLIT_RE.split(answer) if part.strip()]


def looks_like_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text))


def numbers_in(text: str) -> set[str]:
    cleaned = LABEL_RE.sub(" ", text)
    return {n.replace(",", "").rstrip("%") for n in NUMBER_RE.findall(cleaned)}


def validate_answer(answer: str, sources: list[RetrievedChunk]) -> GuardrailReport:
    report = GuardrailReport(passed=True)
    total = len(sources)
    for sentence in split_answer(answer):
        labels = [int(n) for n in LABEL_RE.findall(sentence)]
        if not labels:
            if numbers_in(sentence) or len(sentence.split()) > 4:
                report.uncited_sentences.append(sentence)
            continue
        bad = [n for n in labels if n < 1 or n > total]
        report.invalid_citations.extend(f"S{n}" for n in bad)
        valid = [n for n in labels if 1 <= n <= total]
        cited_text = " ".join(
            f"{sources[n - 1].chunk.text} {sources[n - 1].chunk.para_ref}" for n in valid
        )
        report.unsupported_numbers.extend(sorted(numbers_in(sentence) - numbers_in(cited_text)))
    report.passed = not (
        report.invalid_citations or report.uncited_sentences or report.unsupported_numbers
    )
    return report


def describe_problems(report: GuardrailReport) -> str:
    parts = []
    if report.invalid_citations:
        parts.append(f"invalid citations {report.invalid_citations}")
    if report.uncited_sentences:
        parts.append(f"{len(report.uncited_sentences)} uncited sentence(s)")
    if report.unsupported_numbers:
        parts.append(f"numbers not in the cited source {report.unsupported_numbers}")
    return "; ".join(parts)
