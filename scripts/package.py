"""Build the submission ZIP: code, architecture document and evaluation report.

Excludes local state, virtual environments, caches and downloaded regulator files.
An out-of-date evaluation PDF is left out, so the ZIP cannot carry old scores.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT.parent / "finserv-rca-submission.zip"
SKIP_DIRS = {".rca", ".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
SKIP_FILES = {".coverage"}


def main() -> None:
    report = ROOT / "docs/evaluation/evaluation_report.md"
    pdf = ROOT / "docs/evaluation/Evaluation_Report.pdf"
    stale_pdf = pdf.exists() and report.stat().st_mtime > pdf.stat().st_mtime + 60
    count = 0
    with zipfile.ZipFile(TARGET, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if path.is_dir() or SKIP_DIRS & set(relative.parts) or path.name in SKIP_FILES:
                continue
            if any(part.endswith(".egg-info") for part in relative.parts):
                continue
            if relative.parts[:2] == ("data", "official") and path.name != "provenance.json":
                continue
            if stale_pdf and path == pdf:
                continue
            archive.write(path, Path("finserv-rca") / relative)
            count += 1
    print(f"Wrote {TARGET} with {count} files")
    if stale_pdf:
        print("Left out the old Evaluation_Report.pdf. The current report is the Markdown file.")


if __name__ == "__main__":
    main()
