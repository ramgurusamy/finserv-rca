"""Evaluation runner, ablation study and report writer."""
from __future__ import annotations

import json
import statistics
import time
from datetime import date, datetime, timezone
from pathlib import Path

from ..config import ROOT
from ..llm.providers import GenerationRequest
from ..llm.router import AllProvidersFailed
from ..retrieval.retriever import HybridRetriever
from .metrics import (
    answer_relevance,
    context_precision,
    context_recall,
    fact_coverage,
    faithfulness,
    ref_matches,
    split_claims,
)

GATES = {
    "faithfulness": 0.85,
    "answer_relevance": 0.50,
    "context_precision": 0.70,
    "context_recall": 0.85,
    "fact_coverage": 0.80,
    "refusal_accuracy": 1.00,
}
METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall", "fact_coverage"]
JUDGE_SYSTEM = "Reply with exactly YES or NO. Is the CLAIM fully supported by the CONTEXT?"


def load_golden(path: Path | None = None) -> list[dict]:
    file = path or ROOT / "eval" / "golden.jsonl"
    return [json.loads(line) for line in file.read_text().splitlines() if line.strip()]


def judge_faithfulness(app, answer, contexts) -> float | None:
    """LLM-as-judge variant. It returns None when no LLM provider is configured."""
    if app.settings.llm.generation_chain[0] == "extractive":
        return None
    context = "\n".join(c.chunk.text for c in contexts)
    claims = split_claims(answer)
    if not claims:
        return None
    votes = []
    for claim in claims:
        request = GenerationRequest(
            system=JUDGE_SYSTEM,
            prompt=f"CONTEXT:\n{context}\n\nCLAIM: {claim}\nANSWER:",
            question=claim,
        )
        try:
            result = app.router.generate("generation", request, {"GLOBAL"})
        except AllProvidersFailed:
            return None
        if result.provider == "extractive":
            return None
        votes.append(result.text.strip().upper().startswith("YES"))
    return sum(votes) / len(votes)


def evaluate(app, golden: list[dict], as_of: date | None = None, judge: bool = False) -> dict:
    embedder = app.retriever.embedder
    rows = []
    for item in golden:
        started = time.perf_counter()
        answer = app.rag.ask(item["question"], as_of=as_of)
        elapsed = int((time.perf_counter() - started) * 1000)
        refs = list(dict.fromkeys(c.chunk.para_ref for c in answer.contexts))
        answerable = item.get("answerable", True)
        row = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answerable": answerable,
            "refused": answer.refused,
            "answer": answer.answer,
            "retrieved_refs": refs,
            "gold_refs": item["gold_refs"],
            "provider": answer.provider,
            "model": answer.model,
            "degraded": answer.degraded,
            "latency_ms": elapsed,
            "tokens": answer.tokens,
            "guardrail_notes": answer.guardrails.notes if answer.guardrails else [],
        }
        if answerable:
            row.update(
                context_recall=context_recall(refs, item["gold_refs"]),
                context_precision=context_precision(refs, item["gold_refs"]),
                faithfulness=0.0 if answer.refused else faithfulness(answer.answer, answer.contexts),
                answer_relevance=(
                    0.0 if answer.refused
                    else answer_relevance(item["question"], answer.answer, embedder)
                ),
                fact_coverage=(
                    0.0 if answer.refused else fact_coverage(answer.answer, item["required_facts"])
                ),
            )
            if judge and not answer.refused:
                row["faithfulness_llm"] = judge_faithfulness(app, answer.answer, answer.contexts)
        rows.append(row)
    return summarise(rows)


def summarise(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    scores = {m: statistics.fmean(r[m] for r in scored) for m in METRICS}
    judged = [r["faithfulness_llm"] for r in scored if r.get("faithfulness_llm") is not None]
    if judged:
        scores["faithfulness_llm"] = statistics.fmean(judged)
    scores["refusal_accuracy"] = (
        statistics.fmean(r["refused"] for r in unanswerable) if unanswerable else 1.0
    )
    scores["false_refusal_rate"] = statistics.fmean(r["refused"] for r in scored)
    latencies = sorted(r["latency_ms"] for r in rows)
    return {
        "scores": scores,
        "latency_ms": {
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)],
        },
        "degraded_answers": sum(r["degraded"] for r in rows),
        "rows": rows,
    }


def ablation(app, golden: list[dict]) -> list[dict]:
    """Measure what each retrieval stage adds, on the same questions."""
    cfg = app.settings.retrieval
    original = (cfg.mode, cfg.reranker)
    variants = [
        ("Dense only", "dense", "none"),
        ("BM25 only", "sparse", "none"),
        ("Hybrid (RRF)", "hybrid", "none"),
        (f"Hybrid + rerank ({original[1]})", "hybrid", original[1]),
    ]
    rows = []
    answerable = [g for g in golden if g.get("answerable", True)]
    for label, mode, reranker in variants:
        cfg.mode, cfg.reranker = mode, reranker
        retriever = HybridRetriever(app.settings, app.store)
        recall, precision, hit1 = [], [], []
        for item in answerable:
            refs = list(
                dict.fromkeys(c.chunk.para_ref for c in retriever.retrieve(item["question"]))
            )
            recall.append(context_recall(refs, item["gold_refs"]))
            precision.append(context_precision(refs, item["gold_refs"]))
            hit1.append(bool(refs) and any(ref_matches(g, refs[0]) for g in item["gold_refs"]))
        rows.append(
            {
                "variant": label,
                "context_recall": statistics.fmean(recall),
                "context_precision": statistics.fmean(precision),
                "hit_at_1": statistics.fmean(hit1),
            }
        )
    cfg.mode, cfg.reranker = original
    return rows


def explain_failure(row: dict) -> str:
    if not row["answerable"]:
        return "Answered a question it should have refused."
    if row["refused"]:
        return "Refused a question it should have answered."
    if row["context_recall"] < 1:
        return "Search missed an expected paragraph."
    if row["fact_coverage"] < 1:
        return "Found the right paragraph, but the answer left out the key fact."
    if row["faithfulness"] < 1:
        return "The answer has a claim that no single source supports."
    if row["context_precision"] < 0.5:
        return "Found the right paragraph, but ranked it below less relevant ones."
    return "The answer uses little of the question's wording."


# Plain names for the report. Keys stay the same in results.json.
LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer relevance",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "fact_coverage": "Required facts present",
    "refusal_accuracy": "Unanswerable questions refused",
    "false_refusal_rate": "Answerable questions wrongly refused",
}
TOPICS = {
    "capital": "Capital", "large_exposures": "Large exposures", "mifid": "MiFID II",
    "kyc": "KYC", "psl": "Priority sector", "cross_regulation": "Cross-regulation",
    "unanswerable": "Unanswerable",
}


def topic(category: str) -> str:
    return TOPICS.get(category, category.replace("_", " ").capitalize())


MEANING = {
    "faithfulness": "Each claim in the answer is supported by a source",
    "answer_relevance": "The answer addresses the question",
    "context_precision": "The right paragraphs are ranked near the top",
    "context_recall": "The right paragraphs are found at all",
    "fact_coverage": "The answer states the facts it should",
    "refusal_accuracy": "Out-of-scope questions get a refusal, not a guess",
    "false_refusal_rate": "Lower is better",
}


def failures(results: dict) -> list[dict]:
    selected = []
    for row in results["rows"]:
        if not row["answerable"]:
            failed = not row["refused"]
        else:
            failed = (
                row["refused"]
                or row["context_recall"] < 1
                or row["fact_coverage"] < 1
                or row["faithfulness"] < 1
                or row["context_precision"] < 0.5
            )
        if failed:
            selected.append({**row, "diagnosis": explain_failure(row)})
    return selected


def write_report(app, results: dict, study: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {**results, "ablation": study}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    scores = results["scores"]
    settings = app.settings
    llm = settings.llm.generation_chain[0] != "extractive"
    official = sum(c.content_class == "official" for c in app.store.all_chunks())
    total = len(app.store.all_chunks())
    answer_model = next((r["model"] for r in results["rows"] if r["model"]), "none")
    rows = results["rows"]
    unanswerable = sum(not r["answerable"] for r in rows)
    passed = sum(scores[m] >= g for m, g in GATES.items())
    selected = failures(results)
    best = max(study, key=lambda r: r["context_precision"]) if study else None
    reranked = next((r for r in study if "rerank" in r["variant"].lower()), None)
    hybrid = next((r for r in study if r["variant"].lower().startswith("hybrid (rrf)")), None)

    lines = [
        "# FinServ Compliance Assistant: Evaluation Report",
        "",
        "## Run details",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Date | {datetime.now(timezone.utc):%d %B %Y, %H:%M UTC} |",
        f"| Setup | `{settings.profile}` profile |",
        f"| Answer model | `{answer_model}` |",
        f"| Embedding model | `{app.retriever.embedder.name}` |",
        f"| Re-ranker | `{app.retriever.reranker.name}` |",
        f"| Documents | {total} chunks ({official} from official documents) |",
        f"| Test set | {len(rows)} questions, {unanswerable} of them deliberately unanswerable |",
        "",
        "## Summary",
        "",
        f"- **{passed} of {len(GATES)} quality targets met.**",
    ]
    if not llm:
        lines.append(
            "- **This is the offline baseline.** No LLM was used, and the documents are "
            "paraphrased digests, not official text. Answers are quoted sentences, so "
            "faithfulness is 1.00 by design and says nothing about LLM behaviour. The retrieval "
            "scores are real. Run `make full` on a machine with Ollama for LLM results."
        )
    if best and reranked and hybrid:
        lines.append(
            f"- **Re-ranking did not help.** Best retrieval: {best['variant']} "
            f"(precision {best['context_precision']:.2f}). Adding the re-ranker changed precision "
            f"from {hybrid['context_precision']:.2f} to {reranked['context_precision']:.2f}."
            if reranked["context_precision"] < hybrid["context_precision"]
            else f"- **Best retrieval:** {best['variant']} (precision {best['context_precision']:.2f})."
        )
    lines += [
        f"- **{len(selected)} of {len(rows)} questions need work.** Details are in "
        "\"Questions that need work\" below.",
        "",
        "## Scores",
        "",
        "| Metric | What it checks | Score | Target | Result |",
        "|---|---|---:|---:|---|",
    ]
    for metric, gate in GATES.items():
        value = scores[metric]
        lines.append(
            f"| {LABELS[metric]} | {MEANING[metric]} | {value:.2f} | {gate:.2f} | "
            f"{'Pass' if value >= gate else 'Fail'} |"
        )
    if "faithfulness_llm" in scores:
        lines.append(
            f"| Faithfulness (LLM judge) | A second model checks each claim | "
            f"{scores['faithfulness_llm']:.2f} | n/a | n/a |"
        )
    lines += [
        f"| {LABELS['false_refusal_rate']} | {MEANING['false_refusal_rate']} | "
        f"{scores['false_refusal_rate']:.2f} | n/a | n/a |",
        "",
        f"Response time: median {results['latency_ms']['p50']} ms, p95 "
        f"{results['latency_ms']['p95']} ms. Degraded answers: {results['degraded_answers']}.",
        "",
        "## Retrieval comparison",
        "",
        "Each retrieval step was switched on in turn to see whether it improves results.",
        "",
        "| Retrieval method | Context recall | Context precision | Right paragraph ranked first |",
        "|---|---:|---:|---:|",
    ]
    for row in study:
        lines.append(
            f"| {row['variant']} | {row['context_recall']:.2f} | "
            f"{row['context_precision']:.2f} | {row['hit_at_1']:.2f} |"
        )
    lines += ["", "## Questions that need work", ""]
    if selected:
        lines += ["| ID | Topic | Problem |", "|---|---|---|"]
        lines += [f"| {r['id']} | {topic(r['category'])} | {r['diagnosis']} |" for r in selected]
    else:
        lines.append("None.")
    for row in selected:
        lines += [
            "",
            f"### {row['id']}: {row['question']}",
            "",
            f"- **Problem:** {row['diagnosis']}",
            f"- **Expected paragraphs:** {', '.join(row['gold_refs'])}",
            f"- **Found, in rank order:** {', '.join(row['retrieved_refs'])}",
            f"- **Answer given:** {row['answer']}",
        ]
    lines += [
        "",
        "## Limits of these results",
        "",
        f"- The test set is small ({len(rows)} questions), so one question moves an average by "
        f"about {100 // max(len(rows), 1)} points.",
        "- The refusal threshold and the feature re-ranker were tuned on this same test set, so "
        "the scores are optimistic. A separate test set written by compliance officers is "
        "needed before any release decision.",
        "- Finding the right paragraph is easier in a small corpus with few similar paragraphs. "
        "Expect lower recall on full official documents.",
        "- The answer metrics are automatic checks, not a legal review by a person.",
        "",
        "## Scores by question",
        "",
        "| ID | Topic | Recall | Precision | Faithful | Relevance | Facts |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["answerable"]:
            lines.append(
                f"| {row['id']} | {topic(row['category'])} | {row['context_recall']:.2f} | "
                f"{row['context_precision']:.2f} | {row['faithfulness']:.2f} | "
                f"{row['answer_relevance']:.2f} | {row['fact_coverage']:.2f} |"
            )
        else:
            outcome = "Refused (correct)" if row["refused"] else "Answered (wrong)"
            lines.append(f"| {row['id']} | {topic(row['category'])} | {outcome} | | | | |")
    lines += ["", "## How each metric is calculated", "",
              "- **Context recall:** share of expected paragraphs that were retrieved.",
              "- **Context precision:** how high the expected paragraphs rank (average precision).",
              "- **Faithfulness:** share of answer claims where one source contains at least 60% "
              "of the claim's words and all of its numbers.",
              "- **Answer relevance:** average of question-answer similarity and the share of "
              "question words used in the answer.",
              "- **Required facts present:** share of the labelled facts that appear in the answer.",
              "- **Unanswerable questions refused:** share of out-of-scope questions the system "
              "refused."]
    path = out_dir / "evaluation_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
