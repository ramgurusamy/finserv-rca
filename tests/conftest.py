import pytest

from finserv_rca.config import load_settings
from finserv_rca.container import build


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    base = tmp_path_factory.mktemp("rca")
    cfg = load_settings("offline")
    cfg.store.path = ":memory:"
    cfg.paths.audit_log = str(base / "audit.jsonl")
    cfg.paths.registry = str(base / "registry.json")
    return cfg


@pytest.fixture(scope="session")
def app(settings):
    container = build(settings=settings)
    container.ingestor().ingest_all()
    container.retriever.refresh()
    yield container
    container.close()


@pytest.fixture()
def fresh(tmp_path):
    cfg = load_settings("offline")
    cfg.store.path = ":memory:"
    cfg.paths.audit_log = str(tmp_path / "audit.jsonl")
    cfg.paths.registry = str(tmp_path / "registry.json")
    container = build(settings=cfg)
    container.ingestor().ingest_all()
    container.retriever.refresh()
    yield container
    container.close()
