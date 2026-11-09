"""Audit-committee compliance report for a set of transactions (core scenario 4)."""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

from ..models import Assessment


def build_report(
    assessments: list[Assessment], start: date, end: date, audit_status: tuple[bool, int, str]
) -> tuple[str, dict]:
    ratings = Counter(a.risk_rating.value for a in assessments)
    statuses = Counter(a.status.value for a in assessments)
    frameworks = Counter(f.framework for a in assessments for f in a.findings)
    rules = Counter(f"{f.rule_id}: {f.title}" for a in assessments for f in a.findings)
    open_items = [a for a in assessments if a.clarifying_questions or a.errors]
    ordered = sorted(
        assessments, key=lambda a: ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNDETERMINED"].index(
            a.risk_rating.value
        )
    )
    summary = {
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "transactions_screened": len(assessments),
        "by_rating": dict(ratings),
        "by_status": dict(statuses),
        "by_framework": dict(frameworks),
        "top_findings": rules.most_common(5),
        "audit_chain": {"intact": audit_status[0], "records": audit_status[1]},
    }
    lines = [
        "# Compliance Report for the Internal Audit Committee",
        f"**Period:** {start} to {end}  ",
        f"**Transactions screened:** {len(assessments)}  ",
        f"**Audit chain:** {'intact' if audit_status[0] else 'BROKEN'} "
        f"({audit_status[1]} records)",
        "",
        "> Prepared by an AI-assisted system for compliance officer review. "
        "This is not a legal determination.",
        "",
        "## Posture summary",
        "| Risk rating | Transactions |",
        "|---|---|",
    ]
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNDETERMINED"):
        lines.append(f"| {level} | {ratings.get(level, 0)} |")
    lines += ["", "## Most frequent findings", "| Finding | Count |", "|---|---|"]
    lines += [f"| {name} | {count} |" for name, count in rules.most_common(5)]
    lines += ["", "## Transactions, highest risk first"]
    for item in ordered:
        tx = item.transaction
        lines += [
            f"### {tx.transaction_id or 'Unreferenced'}: {item.risk_rating.value}",
            f"{tx.description}  ",
            f"**Date:** {tx.date or 'not stated'} | **Status:** {item.status.value} | "
            f"**Assessment ID:** `{item.assessment_id}`",
        ]
        if item.required_actions:
            lines.append("**Required actions:**")
            lines += [f"- {a}" for a in item.required_actions]
        if item.citations:
            refs = "; ".join(f"{c.para_ref} (v{c.version})" for c in item.citations)
            lines.append(f"\n**Citations:** {refs}")
        if item.clarifying_questions:
            lines.append("\n**Open questions:**")
            lines += [f"- {q}" for q in item.clarifying_questions]
        lines.append("")
    lines += [
        "## Open items",
        f"{len(open_items)} of {len(assessments)} assessments need more information or "
        "human review before they can be closed.",
    ]
    return "\n".join(lines) + "\n", summary


def write_report(markdown: str, summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compliance_report.md").write_text(markdown)
    (out_dir / "compliance_report.json").write_text(json.dumps(summary, indent=2))
