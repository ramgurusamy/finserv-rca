from finserv_rca.agent.tools import ToolError
from finserv_rca.models import AssessmentStatus, RiskRating


def rules(assessment):
    return {f.rule_id for f in assessment.findings}


def test_cross_border_non_kyc(app):
    a = app.agent.assess("Cross-border payment of $2M to a non-KYC verified entity in a high-risk jurisdiction")
    assert a.risk_rating == RiskRating.CRITICAL
    assert a.transaction.amount == 2_000_000 and a.transaction.currency == "USD"
    assert {"AML-001", "AML-004"} <= rules(a)
    assert a.status == AssessmentStatus.PRELIMINARY_NEEDS_INFO
    assert any("jurisdiction" in q for q in a.clarifying_questions)
    assert a.citations and a.required_actions and a.applicable_regulations


def test_clarification_completes_assessment(app):
    a = app.agent.assess(
        "Cross-border payment of $2M to a non-KYC verified entity in a high-risk jurisdiction",
        clarifications={"jurisdiction": "Myanmar"},
    )
    assert "AML-002" in rules(a) and "AML-004" not in rules(a)
    assert a.status == AssessmentStatus.COMPLETE


def test_intragroup_derivative_needs_measurement(app):
    a = app.agent.assess("Intra-group derivative trade exceeding large exposure threshold")
    assert a.risk_rating == RiskRating.HIGH and "LEX-003" in rules(a)
    assert any("Tier 1" in q for q in a.clarifying_questions)


def test_measured_breach_is_critical_and_cross_referenced(app):
    a = app.agent.assess("Intra-group derivative trade at 30% of Tier 1 capital, amount $5M")
    assert a.risk_rating == RiskRating.CRITICAL and "LEX-001" in rules(a)


def test_exposure_at_limit_is_reportable_not_a_breach(app):
    a = app.agent.assess("Derivative with exposure at 25% of Tier 1 capital")
    assert "LEX-001" not in rules(a) and "LEX-002" in rules(a)


def test_mifid_retail_complex(app):
    a = app.agent.assess("Retail customer investment in a complex product without appropriateness assessment")
    assert a.risk_rating == RiskRating.HIGH and "MIFID-001" in rules(a)
    assert any(c.para_ref == "Art25(3)" for c in a.citations)


def test_nbfc_priority_sector_is_a_bank_obligation(app):
    a = app.agent.assess("NBFC lending transaction requiring priority sector reporting")
    assert "PSL-001" in rules(a) and any(c.para_ref == "PSL-Para 2" for c in a.citations)
    assert a.status == AssessmentStatus.PRELIMINARY_NEEDS_INFO


def test_ceiling_exceeded_raises_rating(app):
    a = app.agent.assess({
        "description": "NBFC lending requiring priority sector reporting",
        "psl_category": "agriculture", "borrower_amount_inr": 1_500_000, "lender_type": "commercial bank",
    })
    assert "PSL-002" in rules(a) and a.risk_rating == RiskRating.HIGH


def test_vague_input_does_not_guess(app):
    a = app.agent.assess("Please check this one")
    assert a.status == AssessmentStatus.INSUFFICIENT_INFO
    assert a.risk_rating == RiskRating.UNDETERMINED and a.clarifying_questions


def test_unknown_kyc_is_not_treated_as_verified(app):
    a = app.agent.assess("Cross-border payment of $10,000 to a company in Germany")
    assert a.transaction.counterparty_kyc_verified is None
    assert any("KYC" in q for q in a.clarifying_questions)


def test_no_rule_clean_case_is_low(app):
    a = app.agent.assess({"description": "Domestic loan", "amount": 100000, "currency": "INR",
                          "instrument_type": "loan", "jurisdiction": "India",
                          "counterparty_kyc_verified": True})
    assert a.risk_rating == RiskRating.LOW and a.status == AssessmentStatus.COMPLETE


def test_lookup_by_transaction_id(app):
    a = app.agent.assess({"transaction_id": "TX-1005", "description": ""})
    assert a.transaction.counterparty == "Entity B"


def test_unknown_transaction_goes_to_human(app):
    a = app.agent.assess({"transaction_id": "TX-404", "description": "Payment of $5M to a non-KYC firm"})
    assert a.status == AssessmentStatus.HUMAN_REVIEW and a.errors


def test_tool_failure_is_handled(fresh, monkeypatch):
    def boom(*args, **kwargs):
        raise ToolError("store unavailable")

    monkeypatch.setattr(fresh.agent.tools, "query_regulations", boom)
    a = fresh.agent.assess("Payment of $2M to a non-KYC entity")
    assert a.status == AssessmentStatus.HUMAN_REVIEW
    assert a.risk_rating == RiskRating.CRITICAL
    assert any("store unavailable" in e for e in a.errors)


def test_every_rule_anchor_exists_in_corpus(app):
    refs = {c.para_ref for c in app.store.all_chunks()}
    missing = [a for r in app.agent.book.rules for a in r["anchors"] if a not in refs]
    assert not missing
