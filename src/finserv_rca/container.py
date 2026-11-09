"""Composition root. Services receive their dependencies here and nowhere else."""
from __future__ import annotations

from dataclasses import dataclass, field

from .audit import AuditLog
from .config import Settings, load_settings
from .ingest.pipeline import Ingestor
from .llm.router import ModelRouter
from .rag import RagService
from .retrieval.embeddings import get_embedder
from .retrieval.retriever import HybridRetriever
from .retrieval.vector_store import VectorStore


@dataclass
class Container:
    settings: Settings
    store: VectorStore
    retriever: HybridRetriever
    router: ModelRouter
    audit: AuditLog
    rag: RagService
    _agent: object = field(default=None, repr=False)

    @property
    def agent(self):
        if self._agent is None:
            from .agent.graph import ComplianceAgent

            self._agent = ComplianceAgent(self)
        return self._agent

    def ingestor(self) -> Ingestor:
        return Ingestor(self.settings, self.store)

    def close(self) -> None:
        self.store.close()


def build(profile: str | None = None, settings: Settings | None = None) -> Container:
    settings = settings or load_settings(profile)
    embedder = get_embedder(settings.embedding.backend, settings.embedding.model)
    store_path = settings.store.path
    if store_path != ":memory:":
        resolved = settings.path(store_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        store_path = str(resolved)
    store = VectorStore(store_path, settings.store.collection, embedder.dim)
    retriever = HybridRetriever(settings, store)
    router = ModelRouter(settings.llm)
    audit = AuditLog(settings.path(settings.paths.audit_log))
    rag = RagService(settings, retriever, router, audit)
    return Container(settings, store, retriever, router, audit, rag)
