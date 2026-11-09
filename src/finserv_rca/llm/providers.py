"""Model providers.

Approach: one OpenAI-compatible client serves Ollama on a laptop and vLLM in
production, so the prototype exercises the production call path. ExtractiveProvider
is the no-LLM fallback: it quotes source sentences and cannot invent text.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import requests

from ..config import ProviderConfig
from ..ingest.chunker import split_sentences
from ..models import RetrievedChunk
from ..retrieval.sparse import tokenize


class ProviderError(RuntimeError):
    """Retryable provider failure. The router moves to the next provider."""


@dataclass
class GenerationRequest:
    system: str
    prompt: str
    question: str = ""
    sources: list[RetrievedChunk] = field(default_factory=list)
    temperature_override: float | None = None


@dataclass
class GenerationResult:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    degraded: bool = False


class Provider(Protocol):
    config: ProviderConfig

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class ExtractiveProvider:
    """Selects the strongest source sentences for the question."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def generate(self, request: GenerationRequest) -> GenerationResult:
        terms = set(tokenize(request.question or request.prompt))
        candidates = []
        top_score = max((item.score for item in request.sources), default=0.0)
        for index, item in enumerate(request.sources, start=1):
            if item.score < 0.75 * top_score:
                continue  # weak sources may support a generator, but are not worth quoting
            for position, sentence in enumerate(split_sentences(item.text)):
                overlap = len(terms & set(tokenize(sentence))) / max(len(terms), 1)
                candidates.append((overlap * (0.5 + item.score), index, position, sentence))
        candidates.sort(key=lambda c: c[0], reverse=True)
        best = candidates[0][0] if candidates else 0.0
        strong = [c for c in candidates[:3] if c[0] >= 0.5 * best]
        picked = sorted(strong, key=lambda c: (c[1], c[2]))
        lines = [f"{sentence.rstrip('.;:')}. [S{index}]" for _, index, _, sentence in picked]
        text = " ".join(lines) if lines else "INSUFFICIENT_EVIDENCE"
        return GenerationResult(text=text, provider=self.config.name, model="extractive-v1")


class OpenAICompatibleProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def generate(self, request: GenerationRequest) -> GenerationResult:
        config = self.config
        headers = {}
        if config.api_key_env:
            headers["Authorization"] = f"Bearer {os.environ.get(config.api_key_env, '')}"
        temperature = (
            config.temperature
            if request.temperature_override is None
            else request.temperature_override
        )
        body = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": temperature,
            "max_tokens": config.max_tokens,
            "stream": False,
        }
        try:
            response = requests.post(
                f"{config.base_url}/chat/completions",
                json=body,
                headers=headers,
                timeout=config.timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"{config.name}: {exc}") from exc
        if not text:
            raise ProviderError(f"{config.name}: empty completion")
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        cost = (
            input_tokens * config.cost_per_1k_input + output_tokens * config.cost_per_1k_output
        ) / 1000
        return GenerationResult(
            text=text,
            provider=config.name,
            model=config.model or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


def build_provider(config: ProviderConfig) -> Provider:
    if config.kind == "extractive":
        return ExtractiveProvider(config)
    return OpenAICompatibleProvider(config)
