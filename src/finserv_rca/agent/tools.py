"""Agent tools. In production these are authenticated, read-only services (see design doc)."""
from __future__ import annotations

import json
from pathlib import Path

from ..models import Citation, RetrievedChunk
from ..rag import to_citation
from ..retrieval.retriever import HybridRetriever


class ToolError(RuntimeError):
    pass


class TransactionStore:
    """Read-only transaction lookup, standing in for the core-banking data API."""

    def __init__(self, path: Path) -> None:
        self.records = {}
        if path.exists():
            self.records = {r["transaction_id"]: r for r in json.loads(path.read_text())}

    def get(self, transaction_id: str) -> dict | None:
        return self.records.get(transaction_id)


class RegulationTools:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def query_regulations(
        self, query: str, frameworks: list[str] | None, k: int = 4
    ) -> list[RetrievedChunk]:
        try:
            return self.retriever.retrieve(query, frameworks=frameworks, k=k)
        except Exception as exc:  # the store boundary: convert to a handled tool failure
            raise ToolError(f"query_regulations failed: {exc}") from exc

    def get_paragraph(self, para_ref: str) -> RetrievedChunk | None:
        """Exact paragraph lookup against current versions, used for rule anchors."""
        base = para_ref.split("(")[0]
        for wanted in (para_ref, base):
            for chunk in self.retriever.sparse.chunks:
                if chunk.para_ref == wanted and chunk.effective_to is None:
                    return RetrievedChunk(chunk=chunk, score=1.0)
        return None

    @staticmethod
    def cite(item: RetrievedChunk, index: int) -> Citation:
        return to_citation(f"S{index}", item)
