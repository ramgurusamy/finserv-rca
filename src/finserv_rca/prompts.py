"""Versioned prompt rendering. The version is recorded on every answer and audit record."""
from __future__ import annotations

from pathlib import Path

import yaml

from .models import RetrievedChunk


class PromptLibrary:
    def __init__(self, path: Path) -> None:
        self.data = yaml.safe_load(path.read_text())
        self.version = self.data["version"]

    @property
    def system(self) -> str:
        return f"{self.data['system']}\n{self.data['reasoning_template']}"

    def render(self, question: str, sources: list[RetrievedChunk], problems: str = "") -> str:
        shots = "\n\n".join(
            f"QUESTION: {s['question']}\nSOURCES:\n{s['sources']}\nANSWER: {s['answer']}"
            for s in self.data["few_shot"]
        )
        listed = "\n".join(
            f"[S{i}] ({s.chunk.framework} {s.chunk.para_ref}, version {s.chunk.version}) {s.text}"
            for i, s in enumerate(sources, start=1)
        )
        prompt = f"EXAMPLES\n{shots}\n\nQUESTION: {question}\nSOURCES:\n{listed}\nANSWER:"
        if problems:
            prompt += "\n" + self.data["strict_retry_suffix"].format(problems=problems)
        return prompt
