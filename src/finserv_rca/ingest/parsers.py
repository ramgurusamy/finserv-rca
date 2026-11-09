"""Multi-format parsing into (para_ref, heading, text) sections.

Approach: regulation is cited by paragraph, so paragraph numbers are the primary
chunk boundary, not character counts. Every format converges on `Section`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader


class ParseError(ValueError):
    """Raised when a document cannot be turned into usable text."""


@dataclass
class Section:
    para_ref: str
    heading: str
    text: str


def read_markdown(path: Path) -> tuple[dict, list[Section]]:
    """Digest format: YAML front matter, then `## <para_ref> <heading>` sections."""
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
    if not match:
        raise ParseError(f"{path.name}: missing YAML front matter")
    meta = yaml.safe_load(match.group(1))
    sections = []
    for block in re.split(r"^## ", match.group(2), flags=re.MULTILINE)[1:]:
        head, _, body = block.partition("\n")
        ref, heading = _split_heading(head.strip())
        if body.strip():
            sections.append(Section(ref, heading, " ".join(body.split())))
    if not sections:
        raise ParseError(f"{path.name}: no sections found")
    return meta, sections


def _split_heading(head: str) -> tuple[str, str]:
    """Split '<ref> <title>' where a ref may contain one space, as in 'KYC-Para 17(3)'."""
    match = re.match(r"^(\S*Para \S+|\S+)\s*(.*)$", head)
    return (match.group(1), match.group(2)) if match else (head, "")


def read_pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise ParseError(f"{path.name}: unreadable PDF ({exc})") from exc
    if len(text.strip()) < 200:
        raise ParseError(f"{path.name}: little or no text layer; route to OCR")
    return text


def read_html_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text("\n")


def split_by_paragraph_numbers(
    text: str, pattern: str, prefix: str = "", default_ref: str = "preamble"
) -> list[Section]:
    """Split official text at numbered paragraphs described by a manifest regex.

    `pattern` must be multi-line anchored with one capture group for the number.
    Text before the first match is kept, so nothing is lost silently.
    """
    regex = re.compile(pattern, re.MULTILINE)
    matches = list(regex.finditer(text))
    if not matches:
        return [Section(default_ref, "", " ".join(text.split()))]
    sections: list[Section] = []
    head = " ".join(text[: matches[0].start()].split())
    if head:
        sections.append(Section(default_ref, "", head))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = " ".join(text[match.end() : end].split())
        if body:
            sections.append(Section(f"{prefix}{match.group(1)}", "", body))
    return sections
