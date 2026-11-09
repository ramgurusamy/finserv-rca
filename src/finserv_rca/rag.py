"""Question answering with citations, abstention, validation and fallback."""
from __future__ import annotations

import re
import time
from datetime import date

from .audit import AuditLog
from .config import Settings
from .guardrails.output import (
    LABEL_RE,
    describe_problems,
    looks_like_injection,
    validate_answer,
)
from .guardrails.pii import redact
from .llm.providers import ExtractiveProvider, GenerationRequest, GenerationResult
from .llm.router import AllProvidersFailed, ModelRouter
from .models import Answer, Citation, GuardrailReport, RetrievedChunk
from .prompts import PromptLibrary
from .retrieval.retriever import HybridRetriever

REFUSAL = "The indexed regulations do not contain enough evidence to answer this question."


def to_citation(label: str, item: RetrievedChunk) -> Citation:
    chunk = item.chunk
    return Citation(
        label=label,
        doc_id=chunk.doc_id,
        para_ref=chunk.para_ref,
        version=chunk.version,
        reference=chunk.reference,
        source_url=chunk.source_url,
        content_class=chunk.content_class,
    )


class RagService:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        router: ModelRouter,
        audit: AuditLog,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.router = router
        self.audit = audit
        self.prompts = PromptLibrary(settings.path(settings.llm.prompt_file))

    def ask(
        self,
        question: str,
        as_of: date | None = None,
        frameworks: list[str] | None = None,
    ) -> Answer:
        started = time.perf_counter()
        clean, pii = redact(question) if self.settings.guardrails.redact_pii else (question, {})
        guard = GuardrailReport(passed=True, pii_redactions=pii)

        if looks_like_injection(clean):
            guard.notes.append("input blocked: prompt-injection pattern")
            return self._finish(clean, REFUSAL, [], guard, started, as_of, refused=True)

        contexts = self.retriever.retrieve(clean, as_of=as_of, frameworks=frameworks)
        threshold = self.settings.retrieval.min_rerank_score
        if not contexts or max(c.strength for c in contexts) < threshold:
            guard.notes.append(f"abstained: evidence strength below {threshold}")
            return self._finish(clean, REFUSAL, contexts, guard, started, as_of, refused=True)

        result, guard = self._generate_validated(clean, contexts, guard)
        if result is None:
            return self._finish(clean, REFUSAL, contexts, guard, started, as_of, refused=True)
        if "INSUFFICIENT_EVIDENCE" in result.text:
            guard.notes.append("model reported insufficient evidence")
            return self._finish(
                clean, REFUSAL, contexts, guard, started, as_of, refused=True, result=result
            )
        return self._finish(clean, result.text, contexts, guard, started, as_of, result=result)

    def _generate_validated(self, question, contexts, guard):
        residencies = {c.chunk.data_residency for c in contexts}
        request = GenerationRequest(
            system=self.prompts.system,
            prompt=self.prompts.render(question, contexts),
            question=question,
            sources=contexts,
        )
        try:
            result = self.router.generate("generation", request, residencies)
        except AllProvidersFailed as exc:
            guard.notes.append(f"all providers failed: {exc}")
            return None, guard

        if "INSUFFICIENT_EVIDENCE" in result.text:
            return result, guard

        report = validate_answer(result.text, contexts)
        if not report.passed and self.settings.guardrails.allow_one_retry:
            guard.notes.append(f"first draft rejected: {describe_problems(report)}")
            request.prompt = self.prompts.render(question, contexts, describe_problems(report))
            try:
                retry = self.router.generate("generation", request, residencies)
                result = retry
                report = validate_answer(result.text, contexts)
            except AllProvidersFailed as exc:
                guard.notes.append(f"retry failed: {exc}")

        if not report.passed:
            # A failing draft is never shown. Quote the sources instead.
            guard.notes.append(f"fell back to extractive: {describe_problems(report)}")
            extractive = next(
                (
                    p
                    for p in self.router.providers.values()
                    if isinstance(p, ExtractiveProvider)
                ),
                None,
            )
            if extractive is None:
                guard.passed = False
                return None, guard
            fallback = extractive.generate(request)
            result = GenerationResult(
                text=fallback.text,
                provider=fallback.provider,
                model=fallback.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                degraded=True,
            )
            report = validate_answer(result.text, contexts)

        report.pii_redactions = guard.pii_redactions
        report.notes = guard.notes + report.notes
        return result, report

    def _finish(
        self, question, text, contexts, guard, started, as_of, refused=False, result=None
    ) -> Answer:
        used = sorted({int(n) for n in LABEL_RE.findall(text)})
        citations = [
            to_citation(f"S{n}", contexts[n - 1]) for n in used if 1 <= n <= len(contexts)
        ]
        tokens = {}
        if result:
            tokens = {"input": result.input_tokens, "output": result.output_tokens}
        answer = Answer(
            question=question,
            answer=re.sub(r"[ \t]+", " ", text).strip(),
            citations=citations,
            grounded=guard.passed and not refused,
            refused=refused,
            degraded=bool(result and result.degraded),
            provider=result.provider if result else "",
            model=result.model if result else "",
            prompt_version=self.prompts.version,
            as_of=as_of,
            guardrails=guard,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tokens=tokens,
            contexts=contexts,
        )
        answer.audit_id = self.audit.append(
            "rag_answer",
            {
                "question": question,
                "as_of": as_of,
                "retrieved": [
                    {
                        "chunk_id": c.chunk.chunk_id,
                        "doc_id": c.chunk.doc_id,
                        "version": c.chunk.version,
                        "para_ref": c.chunk.para_ref,
                        "score": round(c.score, 4),
                    }
                    for c in contexts
                ],
                "answer": answer.answer,
                "model": answer.model,
                "provider": answer.provider,
                "prompt_version": answer.prompt_version,
                "guardrails": guard.model_dump(),
                "refused": refused,
                "degraded": answer.degraded,
                "tokens": tokens,
            },
        )
        return answer
