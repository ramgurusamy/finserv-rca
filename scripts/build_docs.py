"""Assemble the architecture document and render both PDFs with Chromium."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: Calibri, Carlito, Arial, sans-serif; font-size: 11pt; line-height: 1.4; color: #000; background: #fff; }
h1 { font-size: 22pt; margin: 0 0 6pt; }
h2 { font-size: 15pt; border-bottom: 1px solid #000; padding-bottom: 3pt; margin-top: 20pt; }
h3 { font-size: 12pt; margin-top: 14pt; }
h1, h2, h3 { color: #000; }
h2, h3 { break-after: avoid; }
p:has(+ ul), p:has(+ ol), p:has(+ table) { break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt; font-size: 9.5pt; }
tr { break-inside: avoid; }
th { background: #D9D9D9; color: #000; text-align: left; padding: 4pt 5pt; border-bottom: 1px solid #000; }
td { border-bottom: 1px solid #BFBFBF; padding: 3.5pt 5pt; vertical-align: top; }
img { max-width: 100%; max-height: 225mm; display: block; margin: 8pt auto; }
code { font-family: Consolas, Menlo, monospace; font-size: 9pt; background: #F2F2F2; padding: 0 2pt; overflow-wrap: anywhere; }
blockquote { border-left: 3px solid #000; background: #F2F2F2; margin: 8pt 0; padding: 5pt 9pt; }
a { color: #000; overflow-wrap: anywhere; }
"""


def cost_table() -> tuple[str, dict]:
    data = json.loads((DOCS / "cost_model.json").read_text())
    lines = ["| Item | USD a month | Basis |", "|---|---:|---|"]
    lines += [f"| {r['item']} | {r['usd_month']:,} | {r['basis']} |" for r in data["rows"]]
    lines.append(f"| **Total** | **{data['total_usd_month']:,}** | |")
    lines.append(
        f"\nThat is about **${data['cost_per_query_usd']:.3f} a query** across "
        f"{data['queries_month']:,} queries in a {22}-business-day month. "
        f"GPUs are {data['gpu_share']:.0%} of the total."
    )
    return "\n".join(lines), data


def architecture() -> str:
    text = (DOCS / "architecture.md").read_text()
    table, data = cost_table()
    api = data["api_comparison_usd_month"].values()
    adrs = "\n\n".join(p.read_text() for p in sorted((DOCS / "adr").glob("ADR-*.md")))
    return (
        text.replace("{{COST_TABLE}}", table)
        .replace("{{API_LOW}}", f"about ${min(api):,}")
        .replace("{{API_HIGH}}", f"${max(api):,}")
        .replace("{{GPU_COST}}", f"about ${data['self_hosted_gpu_usd_month']:,}")
        .replace("{{ADRS}}", adrs)
        .replace("{{SOURCES}}", (DOCS / "sources.md").read_text())
    )


def render(source: str, base: Path, target: Path, title: str, page) -> None:
    body = markdown.markdown(source, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>"
    temp = base / ".render.html"
    temp.write_text(html)
    page.goto(temp.as_uri())
    page.pdf(
        path=str(target), format="A4", print_background=True, display_header_footer=True,
        header_template="<span></span>",
        footer_template=(
            "<div style='font-size:8px;width:100%;text-align:center;color:#000;font-family:Calibri,Carlito,sans-serif'>"
            f"{title} | Page <span class='pageNumber'></span> of <span class='totalPages'></span></div>"
        ),
        margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
    )
    temp.unlink()


def main() -> None:
    combined = architecture()
    (DOCS / "FinServ_Architecture_Document.md").write_text(combined)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        render(combined, DOCS, DOCS / "FinServ_Architecture_Document.pdf", "FinServ Compliance Assistant: Architecture", page)
        evaluation = DOCS / "evaluation"
        render(
            (evaluation / "evaluation_report.md").read_text(), evaluation,
            evaluation / "Evaluation_Report.pdf", "FinServ Compliance Assistant: Evaluation", page,
        )
        browser.close()
    print("Built PDFs", file=sys.stderr)


if __name__ == "__main__":
    main()
