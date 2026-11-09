"""Task routing with residency enforcement, budget control and circuit breaking.

Order of checks for each provider in a chain:
residency policy -> daily budget -> circuit breaker -> call.
A policy denial is never retried against the same provider and is always logged.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..config import LLMConfig, ProviderConfig
from .providers import (
    GenerationRequest,
    GenerationResult,
    Provider,
    ProviderError,
    build_provider,
)

log = logging.getLogger(__name__)


class AllProvidersFailed(RuntimeError):
    pass


def residency_allowed(provider: ProviderConfig, residencies: set[str]) -> bool:
    """Self-hosted endpoints must still be cleared for the data's residency class."""
    return residencies.issubset(set(provider.allowed_residency))


@dataclass
class Breaker:
    failures: int = 0
    open_until: float = 0.0


class ModelRouter:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.providers: dict[str, Provider] = {p.name: build_provider(p) for p in config.providers}
        self.breakers: dict[str, Breaker] = {p.name: Breaker() for p in config.providers}
        self.spent_usd = 0.0
        self.denials: list[str] = []

    def _chain(self, task: str) -> list[str]:
        return (
            self.config.classification_chain
            if task == "classification"
            else self.config.generation_chain
        )

    def generate(
        self, task: str, request: GenerationRequest, residencies: set[str]
    ) -> GenerationResult:
        chain = self._chain(task)
        errors: list[str] = []
        for position, name in enumerate(chain):
            provider = self.providers[name]
            config = provider.config
            if not residency_allowed(config, residencies):
                reason = f"{name}: residency denied for {sorted(residencies)}"
                self.denials.append(reason)
                errors.append(reason)
                log.warning(reason)
                continue
            if config.hosting == "external" and self.spent_usd >= self.config.daily_budget_usd:
                errors.append(f"{name}: daily budget reached")
                continue
            breaker = self.breakers[name]
            if time.monotonic() < breaker.open_until:
                errors.append(f"{name}: circuit open")
                continue
            try:
                result = provider.generate(request)
            except ProviderError as exc:
                breaker.failures += 1
                if breaker.failures >= self.config.breaker_failure_threshold:
                    breaker.open_until = time.monotonic() + self.config.breaker_cooldown_s
                errors.append(str(exc))
                log.warning("Provider failed: %s", exc)
                continue
            breaker.failures = 0
            self.spent_usd += result.cost_usd
            result.degraded = position > 0
            return result
        raise AllProvidersFailed("; ".join(errors))
