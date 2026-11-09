"""Transaction parsing and rule evaluation. Both are deterministic and unit-testable."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..models import Transaction

MULTIPLIERS = {
    "k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6,
    "bn": 1e9, "billion": 1e9, "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7,
}
AMOUNT_RE = re.compile(
    r"(?P<cur>\$|USD|EUR|€|INR|Rs\.?|₹)\s?(?P<num>\d[\d,]*(?:\.\d+)?)\s?"
    r"(?P<mult>k|thousand|mn|m|million|bn|billion|lakhs?|crores?)?\b",
    re.IGNORECASE,
)
CURRENCY = {"$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR", "inr": "INR",
            "rs": "INR", "rs.": "INR", "₹": "INR"}


class RuleBook:
    def __init__(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text())
        self.version: str = data["version"]
        self.lists: dict[str, list[str]] = data["lists"]
        self.rules: list[dict[str, Any]] = data["rules"]
        self.questions: dict[str, str] = data["questions"]

    def _matches(self, condition: dict, tx: Transaction) -> bool:
        value = getattr(tx, condition["field"])
        op, target = condition["op"], condition.get("value")
        if op == "is_none":
            return value is None
        if value is None:
            return False
        if op == "eq":
            return value == target
        if op == "gte":
            return value >= target
        if op == "gt":
            return value > target
        if op == "lte":
            return value <= target
        if op == "in_list":
            return str(value).lower() in self.lists[target]
        raise ValueError(f"Unknown operator: {op}")

    def evaluate(self, tx: Transaction) -> list[dict[str, Any]]:
        return [r for r in self.rules if all(self._matches(c, tx) for c in r["when"])]

    def find_jurisdiction(self, text: str) -> str | None:
        lowered = text.lower()
        for name in sorted(
            self.lists["call_for_action"] + self.lists["increased_monitoring"],
            key=len,
            reverse=True,
        ):
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                return name
        match = re.search(r"\bto (?:an? )?[\w\s-]{0,30}?\bin ([A-Z][a-z]+(?: [A-Z][a-z]+)?)\b", text)
        return match.group(1) if match else None


def parse_amount(text: str) -> tuple[float | None, str | None]:
    match = AMOUNT_RE.search(text)
    if not match:
        return None, None
    value = float(match.group("num").replace(",", ""))
    multiplier = MULTIPLIERS.get((match.group("mult") or "").lower(), 1)
    return value * multiplier, CURRENCY[match.group("cur").lower()]


def parse_description(text: str, book: RuleBook) -> dict[str, Any]:
    """Conservative extraction: anything not clearly stated stays None."""
    low = text.lower()
    fields: dict[str, Any] = {}
    amount, currency = parse_amount(text)
    if amount is not None:
        fields.update(amount=amount, currency=currency)

    if re.search(r"non[- ]kyc|not kyc|without kyc|kyc (?:is )?(?:not|pending|incomplete)|unverified", low):
        fields["counterparty_kyc_verified"] = False
    elif re.search(r"kyc[- ]verified|kyc (?:is )?complete", low):
        fields["counterparty_kyc_verified"] = True

    if "high-risk jurisdiction" in low or "high risk jurisdiction" in low:
        fields["jurisdiction_risk_stated"] = True
    jurisdiction = book.find_jurisdiction(text)
    if jurisdiction:
        fields["jurisdiction"] = jurisdiction
    if re.search(r"cross[- ]border|international|overseas|offshore", low):
        fields["cross_border"] = True
    if re.search(r"intra[- ]?group", low):
        fields["intra_group"] = True
    if re.search(r"(?:exceed\w*|above|over|breach\w*) (?:the )?large exposure", low):
        fields["exceeds_large_exposure_stated"] = True

    pct = re.search(r"(\d+(?:\.\d+)?)\s?%\s+of\s+(?:its\s+|our\s+|the\s+)?tier\s?1", low)
    if pct:
        fields["exposure_pct_tier1"] = float(pct.group(1))

    if re.search(r"\bderivative|swap|forward|option\b", low):
        fields["instrument_type"] = "derivative"
    elif re.search(r"\bpayment|remittance|wire|transfer\b", low):
        fields["instrument_type"] = "payment"
    elif re.search(r"\blending|loan\b", low):
        fields["instrument_type"] = "loan"
    elif re.search(r"\binvestment|structured product|complex product\b", low):
        fields["instrument_type"] = "investment"

    if "retail" in low:
        fields["client_category"] = "retail"
    elif re.search(r"professional client|eligible counterparty", low):
        fields["client_category"] = "professional"
    if re.search(r"complex product|structured product|\bcfd\b|complex instrument", low):
        fields["complex_product"] = True
    if re.search(r"without (?:an? )?appropriateness|no appropriateness|appropriateness (?:not|missing)", low):
        fields["appropriateness_done"] = False
    elif re.search(r"appropriateness (?:assessment )?(?:done|completed|passed)", low):
        fields["appropriateness_done"] = True

    if "nbfc" in low:
        fields["nbfc_involved"] = True
    if re.search(r"priority sector|\bpsl\b", low):
        fields["psl_relevant"] = True
    if "agricultur" in low:
        fields["psl_category"] = "agriculture"
    elif re.search(r"\bmse\b|micro and small|\bmsme\b", low):
        fields["psl_category"] = "mse"
    return fields
