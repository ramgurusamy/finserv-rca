"""Config-driven behaviour. Profiles live in config/*.yaml; nothing is hard-coded in services."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class IngestConfig(BaseModel):
    corpus_dir: str = "data/corpus"
    official_dir: str = "data/official"
    sources_manifest: str = "data/sources.yaml"
    use_official: bool = False
    max_chunk_tokens: int = 220
    overlap_tokens: int = 30


class EmbeddingConfig(BaseModel):
    backend: Literal["wordllama", "fastembed"] = "wordllama"
    model: str = "BAAI/bge-small-en-v1.5"


class StoreConfig(BaseModel):
    path: str = ".rca/qdrant"
    collection: str = "regulatory_chunks"


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    dense_k: int = 30
    sparse_k: int = 30
    rrf_k: int = 60
    rerank_top_n: int = 20
    final_k: int = 6
    min_rerank_score: float = 0.23
    reranker: Literal["none", "feature", "cross_encoder"] = "feature"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    compression_max_sentences: int = 4
    glossary: dict[str, str] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    name: str
    kind: Literal["extractive", "openai_compatible"]
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    hosting: Literal["self_hosted", "external"] = "self_hosted"
    region: str = "LOCAL"
    allowed_residency: list[str] = Field(default_factory=lambda: ["IN", "EU", "GLOBAL"])
    timeout_s: int = 240
    max_tokens: int = 350
    temperature: float = 0.0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


class LLMConfig(BaseModel):
    prompt_file: str = "config/prompts.yaml"
    generation_chain: list[str] = Field(default_factory=lambda: ["extractive"])
    classification_chain: list[str] = Field(default_factory=lambda: ["extractive"])
    daily_budget_usd: float = 25.0
    breaker_failure_threshold: int = 3
    breaker_cooldown_s: int = 60
    providers: list[ProviderConfig] = Field(
        default_factory=lambda: [ProviderConfig(name="extractive", kind="extractive")]
    )


class GuardrailConfig(BaseModel):
    require_citations: bool = True
    allow_one_retry: bool = True
    redact_pii: bool = True


class AgentConfig(BaseModel):
    rules_file: str = "data/policies/rules.yaml"
    retrieval_retries: int = 1
    usd_inr_rate_illustrative: float = 88.0


class PathsConfig(BaseModel):
    audit_log: str = ".rca/audit.jsonl"
    registry: str = ".rca/registry.json"
    policies: str = "data/policies/internal_policies.yaml"
    transactions: str = "data/transactions/sample_batch.json"


class Settings(BaseModel):
    profile: str = "offline"
    ingest: IngestConfig = IngestConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    store: StoreConfig = StoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    guardrails: GuardrailConfig = GuardrailConfig()
    agent: AgentConfig = AgentConfig()
    paths: PathsConfig = PathsConfig()

    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else ROOT / p


def load_settings(profile: str | None = None) -> Settings:
    """Load config/<profile>.yaml. RCA_PROFILE selects the default profile."""
    name = profile or os.environ.get("RCA_PROFILE", "offline")
    file = ROOT / "config" / f"{name}.yaml"
    if not file.exists():
        raise FileNotFoundError(f"Unknown profile '{name}': {file} does not exist")
    data = yaml.safe_load(file.read_text()) or {}
    settings = Settings.model_validate({**data, "profile": name})
    override = os.environ.get("RCA_OLLAMA_MODEL")
    if override:
        for provider in settings.llm.providers:
            if provider.base_url and "11434" in provider.base_url:
                provider.model = override
    return settings
