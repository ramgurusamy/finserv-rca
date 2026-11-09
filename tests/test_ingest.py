from datetime import date
from pathlib import Path

import pytest

from finserv_rca.ingest.chunker import window_section
from finserv_rca.ingest.parsers import (
    ParseError,
    read_markdown,
    read_pdf_text,
    split_by_paragraph_numbers,
)
from finserv_rca.retrieval.vector_store import VectorStore

DEMO = Path(__file__).parents[1] / "data/demo/rbi-demo-circular.md"


def test_at_least_five_documents(app):
    assert len({c.doc_id for c in app.store.all_chunks()}) >= 5


def test_reingest_is_idempotent(fresh):
    before = len(fresh.store.all_chunks())
    result = fresh.ingestor().ingest_all()
    assert result.added == 0 and result.superseded == 0
    assert len(fresh.store.all_chunks()) == before


def test_amendment_supersedes_only_touched_paragraphs(fresh):
    result = fresh.ingestor().ingest_file(DEMO)
    assert result.superseded == 2
    fresh.retriever.refresh()
    current = fresh.retriever.retrieve("retain transaction records", as_of=date(2027, 2, 1))
    assert any("eight years" in c.chunk.text for c in current)
    untouched = [c for c in fresh.store.all_chunks() if c.para_ref == "KYC-Para 17(3)"]
    assert len(untouched) == 1 and untouched[0].effective_to is None


def test_as_of_query_returns_the_older_version(fresh):
    fresh.ingestor().ingest_file(DEMO)
    fresh.retriever.refresh()
    past = fresh.retriever.retrieve("retain transaction records", as_of=date(2026, 11, 1))
    texts = " ".join(c.chunk.text for c in past)
    assert "five years" in texts and "eight years" not in texts


def test_long_paragraph_is_windowed_with_overlap():
    sentence = "This is one full sentence about capital rules and limits. "
    windows = window_section(sentence * 30, max_tokens=40, overlap=12)
    assert len(windows) > 1
    assert windows[0].split(". ")[-1].strip(". ") == windows[1].split(". ")[0].strip(". ")


def test_paragraph_splitter_keeps_preamble():
    text = "Intro text\n20.1 First para\n20.2 Second para"
    sections = split_by_paragraph_numbers(text, r"^(\d{2}\.\d{1,3})\s", "RBC")
    assert [s.para_ref for s in sections] == ["preamble", "RBC20.1", "RBC20.2"]


def test_pdf_ingestion_path(tmp_path):
    from reportlab.pdfgen import canvas

    path = tmp_path / "sample.pdf"
    pdf = canvas.Canvas(str(path))
    for row, line in enumerate(["20.1 Banks must hold capital of at least the stated minimum."] * 12):
        pdf.drawString(40, 780 - row * 14, line)
    pdf.save()
    assert "capital" in read_pdf_text(path)


def test_scanned_pdf_is_routed_to_ocr(tmp_path):
    from reportlab.pdfgen import canvas

    path = tmp_path / "blank.pdf"
    pdf = canvas.Canvas(str(path))
    pdf.showPage()
    pdf.save()
    with pytest.raises(ParseError, match="OCR"):
        read_pdf_text(path)


def test_malformed_markdown_raises(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text("no front matter")
    with pytest.raises(ParseError):
        read_markdown(path)


def test_store_filter_excludes_future_versions():
    store = VectorStore(":memory:", "t", 3)
    assert store._filter(date(2026, 1, 1), ["RBI"]).must
