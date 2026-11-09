import json
from datetime import date
from pathlib import Path

from finserv_rca.audit import AuditLog
from finserv_rca.eval.metrics import (
    context_precision,
    context_recall,
    fact_coverage,
    faithfulness,
    ref_matches,
)
from finserv_rca.eval.runner import evaluate, load_golden
from finserv_rca.features.impact import analyse_change
from finserv_rca.features.report import build_report
from finserv_rca.models import Chunk, RetrievedChunk

ROOT = Path(__file__).parents[1]
DEMO = ROOT / "data/demo/rbi-demo-circular.md"


def test_impact_finds_policies_rules_and_number_change(fresh):
    ingestor = fresh.ingestor()
    result = ingestor.ingest_file(DEMO)
    chunks = [c for c in fresh.store.all_chunks() if c.version == "2027-01-01"]
    report = analyse_change(
        result.changed_refs, chunks, ROOT / "data/policies/internal_policies.yaml",
        ROOT / "data/policies/rules.yaml", ingestor.embedder,
    )
    assert {p["id"] for p in report["directly_affected_policies"]} == {"POL-AML-02", "POL-AML-03"}
    assert {"AML-002", "AML-003"} <= set(report["affected_screening_rules"])
    assert "payment" in report["affected_transaction_types"]


def test_report_has_required_sections(app):
    rows = json.loads((ROOT / "data/transactions/sample_batch.json").read_text())
    assessments = [app.agent.assess(r) for r in rows]
    markdown, summary = build_report(assessments, date(2026, 10, 1), date(2026, 10, 31), (True, 8, "ok"))
    assert summary["transactions_screened"] == 8
    assert "CRITICAL" in markdown and "Assessment ID" in markdown and "Open items" in markdown


def test_audit_chain_detects_tampering(tmp_path):
    path = tmp_path / "a.jsonl"
    log = AuditLog(path)
    log.append("x", {"v": 1})
    log.append("x", {"v": 2})
    assert log.verify()[0]
    path.write_text(path.read_text().replace('"v": 1', '"v": 9'))
    assert not log.verify()[0]


def test_metric_arithmetic():
    assert context_recall(["a", "b"], ["a", "c"]) == 0.5
    assert context_precision(["x", "a"], ["a"]) == 0.5
    assert fact_coverage("the limit is 25%", ["25%", "tier 1"]) == 0.5
    assert ref_matches("KYC-Para 17(3)", "KYC-Para 17")
    assert not ref_matches("KYC-Para 17(3)", "KYC-Para 1")


def test_faithfulness_catches_unsupported_claim():
    chunk = Chunk(chunk_id="a" * 32, doc_id="d", title="t", framework="f", jurisdiction="j",
                  data_residency="GLOBAL", version="1", effective_from="2023-01-01", para_ref="p",
                  text="Tier 1 capital must be at least 6% of RWA.", source_url="u")
    item = RetrievedChunk(chunk=chunk, score=1)
    assert faithfulness("Tier 1 capital must be at least 6% of RWA. [S1]", [item]) == 1
    assert faithfulness("Tier 1 capital must be at least 9% of RWA. [S1]", [item]) == 0


def test_golden_dataset_shape():
    golden = load_golden()
    assert 15 <= len(golden) <= 20
    assert all(g["ground_truth"] for g in golden)


def test_evaluation_runs_end_to_end(app):
    results = evaluate(app, load_golden())
    assert results["scores"]["context_recall"] >= 0.85
    assert results["scores"]["refusal_accuracy"] == 1.0


def test_laptop_profile_is_valid():
    from finserv_rca.config import load_settings

    settings = load_settings("laptop")
    assert settings.llm.generation_chain == ["ollama-local", "extractive"]
    names = {p.name for p in settings.llm.providers}
    assert set(settings.llm.generation_chain) <= names
