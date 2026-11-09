"""Regulatory change impact analysis (core scenario 3).

Direct dependencies come from the policy register and rule anchors. Semantic
matches only add 'possible' items for a person to confirm.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

from ..guardrails.output import numbers_in
from ..models import Chunk

POSSIBLE_THRESHOLD = 0.35
WORD_NUMBERS = dict(
    zip(["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"], ["2", "3", "4", "5", "6", "7", "8", "9", "10"])
)


def stated_numbers(text: str) -> set[str]:
    """Digits and spelled-out numbers, so 'five years' to 'eight years' is detected."""
    words = {WORD_NUMBERS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in WORD_NUMBERS}
    return numbers_in(text) | words


def analyse_change(
    changes: list[dict],
    new_chunks: list[Chunk],
    policy_file: Path,
    rule_file: Path,
    embedder,
) -> dict:
    policies = yaml.safe_load(policy_file.read_text())["policies"]
    rules = yaml.safe_load(rule_file.read_text())["rules"]
    changed_refs = {c["para_ref"] for c in changes}
    new_by_ref = {c.para_ref: c for c in new_chunks}

    change_rows = []
    for change in changes:
        new = new_by_ref.get(change["para_ref"])
        new_text = new.text if new else ""
        old_numbers = stated_numbers(change["old_text"])
        new_numbers = stated_numbers(new_text)
        change_rows.append(
            {
                **{k: change[k] for k in ("doc_id", "para_ref", "old_version", "new_version")},
                "numbers_removed": sorted(old_numbers - new_numbers),
                "numbers_added": sorted(new_numbers - old_numbers),
                "new_text": new_text,
            }
        )

    direct = []
    for policy in policies:
        hit = sorted(changed_refs & set(policy["depends_on"]))
        if hit:
            direct.append({**_policy_view(policy), "changed_paragraphs": hit})

    possible = []
    direct_ids = {p["id"] for p in direct}
    text = " ".join(c.text for c in new_chunks if c.para_ref in changed_refs)
    remaining = [p for p in policies if p["id"] not in direct_ids]
    if text and remaining:
        vectors = embedder.embed([text] + [f"{p['title']}. {p['description']}" for p in remaining])
        scores = vectors[1:] @ vectors[0]
        for policy, score in zip(remaining, np.atleast_1d(scores)):
            if score >= POSSIBLE_THRESHOLD:
                possible.append({**_policy_view(policy), "similarity": round(float(score), 3)})

    affected_rules = sorted(r["id"] for r in rules if changed_refs & set(r["anchors"]))
    types = sorted({t for p in direct for t in p["transaction_types"]})
    return {
        "changes": change_rows,
        "directly_affected_policies": direct,
        "possibly_affected_policies": possible,
        "affected_screening_rules": affected_rules,
        "affected_transaction_types": types,
    }


def _policy_view(policy: dict) -> dict:
    return {k: policy[k] for k in ("id", "title", "owner", "transaction_types")}
