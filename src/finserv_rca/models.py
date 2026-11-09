"""Shared domain types. Every layer exchanges these, never raw dicts."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    framework: str
    jurisdiction: str
    data_residency: str
    version: str
    effective_from: date
    effective_to: date | None = None
    para_ref: str
    heading: str = ""
    text: str
    source_url: str
    reference: str = ""
    content_class: str = "digest"
    content_hash: str = ""


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    compressed_text: str | None = None
    evidence: float | None = None  # query-evidence strength, used by the abstain gate

    @property
    def strength(self) -> float:
        return self.score if self.evidence is None else self.evidence

    @property
    def text(self) -> str:
        return self.compressed_text or self.chunk.text


class Citation(BaseModel):
    label: str
    doc_id: str
    para_ref: str
    version: str
    reference: str
    source_url: str
    content_class: str


class GuardrailReport(BaseModel):
    passed: bool
    invalid_citations: list[str] = Field(default_factory=list)
    uncited_sentences: list[str] = Field(default_factory=list)
    unsupported_numbers: list[str] = Field(default_factory=list)
    pii_redactions: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
    refused: bool = False
    degraded: bool = False
    provider: str = ""
    model: str = ""
    prompt_version: str = ""
    as_of: date | None = None
    guardrails: GuardrailReport | None = None
    latency_ms: int = 0
    tokens: dict[str, int] = Field(default_factory=dict)
    audit_id: str = ""
    contexts: list[RetrievedChunk] = Field(default_factory=list, exclude=True)


class RiskRating(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNDETERMINED = "UNDETERMINED"


RISK_ORDER = {
    RiskRating.UNDETERMINED: 0,
    RiskRating.LOW: 1,
    RiskRating.MEDIUM: 2,
    RiskRating.HIGH: 3,
    RiskRating.CRITICAL: 4,
}


class AssessmentStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PRELIMINARY_NEEDS_INFO = "PRELIMINARY_NEEDS_INFO"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class Transaction(BaseModel):
    """Normalised transaction. None means unknown, which is different from False."""

    transaction_id: str | None = None
    description: str = ""
    amount: float | None = None
    currency: str | None = None
    instrument_type: str | None = None
    counterparty: str | None = None
    counterparty_kyc_verified: bool | None = None
    jurisdiction: str | None = None
    jurisdiction_risk_stated: bool | None = None
    cross_border: bool | None = None
    intra_group: bool | None = None
    exposure_pct_tier1: float | None = None
    exceeds_large_exposure_stated: bool | None = None
    client_category: str | None = None
    complex_product: bool | None = None
    appropriateness_done: bool | None = None
    lender_type: str | None = None
    nbfc_involved: bool | None = None
    psl_relevant: bool | None = None
    borrower_amount_inr: float | None = None
    psl_category: str | None = None
    date: str | None = None


class Finding(BaseModel):
    rule_id: str
    title: str
    framework: str
    severity: RiskRating
    rationale: str
    required_actions: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence_found: bool = False


class Assessment(BaseModel):
    assessment_id: str
    status: AssessmentStatus
    risk_rating: RiskRating
    summary: str
    transaction: Transaction
    applicable_regulations: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    cross_regulation_notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    human_review_required: bool = True
    disclaimer: str = (
        "Decision support for a compliance officer. Not a legal determination."
    )
    trace: list[dict[str, Any]] = Field(default_factory=list)
