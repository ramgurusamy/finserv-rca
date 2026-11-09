"""Command-line entry point."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import typer

from .config import ROOT
from .container import build

app = typer.Typer(no_args_is_help=True, add_completion=False)
PROFILE = typer.Option(None, "--profile", "-p", help="Config profile: offline or laptop")


@app.command()
def ingest(profile: str | None = PROFILE) -> None:
    """Parse, chunk, embed and version the corpus."""
    container = build(profile)
    result = container.ingestor().ingest_all()
    container.close()
    typer.echo(
        f"documents={result.documents} added={result.added} unchanged={result.unchanged} "
        f"superseded={result.superseded} failed={result.failed}"
    )


@app.command()
def ask(
    question: str,
    as_of: str | None = typer.Option(None, help="Answer as of a date, YYYY-MM-DD"),
    profile: str | None = PROFILE,
) -> None:
    """Ask a regulatory question and get a cited answer."""
    container = build(profile)
    answer = container.rag.ask(question, as_of=date.fromisoformat(as_of) if as_of else None)
    typer.echo(answer.model_dump_json(indent=2))
    container.close()


@app.command()
def screen(
    description: str | None = typer.Argument(None),
    file: Path | None = typer.Option(None, help="JSON transaction payload"),
    profile: str | None = PROFILE,
) -> None:
    """Screen one transaction and print a structured assessment."""
    payload = json.loads(file.read_text()) if file else description
    if not payload:
        raise typer.BadParameter("Provide a description or --file")
    container = build(profile)
    typer.echo(container.agent.assess(payload).model_dump_json(indent=2, exclude={"trace"}))
    container.close()


@app.command()
def impact(
    circular: Path = typer.Argument(ROOT / "data/demo/rbi-demo-circular.md"),
    profile: str | None = PROFILE,
    apply: bool = typer.Option(False, help="Keep the change in the store"),
) -> None:
    """Show which policies, rules and transaction types a circular affects."""
    from .features.impact import analyse_change
    from .ingest.parsers import read_markdown

    settings_container = build(profile)
    settings = settings_container.settings
    if not apply:
        # Work on a throwaway in-memory copy, so a preview never changes the real store.
        settings_container.close()
        settings.store.path = ":memory:"
        settings_container = build(settings=settings)
        settings_container.ingestor().ingest_all()
    ingestor = settings_container.ingestor()
    result = ingestor.ingest_file(circular)
    meta, _ = read_markdown(circular)
    new_chunks = [
        c
        for c in settings_container.store.all_chunks()
        if c.doc_id == meta["doc_id"] and c.version == str(meta["version"])
    ]
    report = analyse_change(
        result.changed_refs,
        new_chunks,
        settings.path(settings.paths.policies),
        settings.path(settings.agent.rules_file),
        ingestor.embedder,
    )
    settings_container.audit.append(
        "impact_analysis", {"circular": circular.name, "summary": report}
    )
    typer.echo(json.dumps(report, indent=2))
    settings_container.close()


@app.command()
def report(
    start: str = typer.Option("2026-10-01"),
    end: str = typer.Option("2026-10-31"),
    out: Path = typer.Option(ROOT / "sample_outputs"),
    profile: str | None = PROFILE,
) -> None:
    """Generate the audit-committee report for a transaction batch."""
    from .features.report import build_report, write_report

    container = build(profile)
    rows = json.loads(container.settings.path(container.settings.paths.transactions).read_text())
    begin, finish = date.fromisoformat(start), date.fromisoformat(end)
    rows = [r for r in rows if begin <= date.fromisoformat(r["date"]) <= finish]
    assessments = [container.agent.assess(row) for row in rows]
    markdown, summary = build_report(assessments, begin, finish, container.audit.verify())
    write_report(markdown, summary, out)
    container.close()
    typer.echo(f"Wrote {out / 'compliance_report.md'}")


@app.command()
def evaluate(
    profile: str | None = PROFILE,
    judge: bool = typer.Option(False, help="Add LLM-as-judge faithfulness"),
    out: Path = typer.Option(ROOT / "docs/evaluation"),
) -> None:
    """Run the golden dataset and write the evaluation report."""
    from .eval.runner import ablation, load_golden, write_report
    from .eval.runner import evaluate as run

    container = build(profile)
    golden = load_golden()
    results = run(container, golden, judge=judge)
    study = ablation(container, golden)
    path = write_report(container, results, study, out)
    container.close()
    typer.echo(json.dumps(results["scores"], indent=2))
    typer.echo(f"Wrote {path}")


@app.command("audit-verify")
def audit_verify(profile: str | None = PROFILE) -> None:
    """Check the audit log's hash chain."""
    container = build(profile)
    ok, count, detail = container.audit.verify()
    container.close()
    typer.echo(f"ok={ok} records={count} detail={detail}")
    raise typer.Exit(0 if ok else 1)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    app()


if __name__ == "__main__":
    main()
