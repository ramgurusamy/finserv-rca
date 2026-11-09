"""PII redaction applied before prompts, logs and audit records.

Order matters: card numbers run first, so a 16-digit card is not read as Aadhaar.
Patterns cover India, EU and US identifiers plus cards and contact data. This is a
deterministic first layer; production adds an NER model (Presidio or GLiNER).
"""
from __future__ import annotations

import re

PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PAN_IN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "AADHAAR_IN": re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b"),
    "SSN_US": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b"),
    "PHONE": re.compile(r"(?<![\d.])\+\d{1,3}[ -]?\d{6,12}\b"),
    "ACCOUNT": re.compile(r"(?i)\b(?:account|a/c)\s*(?:no\.?|number)?[:\s#]*\d{6,18}\b"),
}


def redact(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for name, pattern in PATTERNS.items():
        text, hits = pattern.subn(f"[{name}]", text)
        if hits:
            counts[name] = hits
    return text, counts
