"""LangGraph compliance agent.

Approach: an explicit state machine, not a free-running ReAct loop, because an
auditor must be able to replay exactly what ran. Rules set the rating. Retrieval
supplies evidence. The LLM may only draft the narrative, and that draft must pass
the same guardrails as any other answer. If it fails, a template is used instead.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..guardrails.output import validate_answer
from ..guardrails.pii import redact
from ..llm.providers import GenerationRequest
from ..llm.router import AllProvidersFailed
from ..models import (
    RISK_ORDER,
    Assessment,
    AssessmentStatus,
    Citation,
    Finding,
    RetrievedChunk,
    RiskRating,
    Transaction,
)
from .rules import RuleBook, parse_description
from .tools import RegulationTools, ToolError, TransactionStore

log = logging.getLogger(__name__)

NARRATIVE_SYSTEM = (
    "You write the summary of a compliance assessment. Use only the FINDINGS and SOURCES. "
    "Do not change the risk rating. End every sentence with a source label such as [S1]. "
    "Copy numbers exactly. Write at most three sentences."
)


class AgentState(TypedDict, total=False):
    raw: Any
    clarifications: dict[str, Any]
    transaction: Transaction
    trace: list[dict[str, Any]]
    errors: list[str]
    triggered: list[dict[str, Any]]
    evidence: dict[str, list[RetrievedChunk]]
    gaps: list[str]
    attempts: int
    findings: list[Finding]
    questions: list[str]
    cross_notes: list[str]
    summary: str
    status: AssessmentStatus
    rating: RiskRating
    assessment: Assessment


class ComplianceAgent:
    def __init__(self, app) -> None:
        self.app = app
        self.book = RuleBook(app.settings.path(app.settings.agent.rules_file))
        self.tools = RegulationTools(app.retriever)
        self.transactions = TransactionStore(app.settings.path(app.settings.paths.transactions))
        self.graph = self._build()

    def _build(self):
        graph = StateGraph(AgentState)
        for name in (
            "intake", "pull_transaction", "check_sufficiency", "screen", "retrieve",
            "broaden", "cross_reference", "assess", "validate", "finalize",
        ):
            graph.add_node(name, getattr(self, f"_{name}"))
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "pull_transaction")
        graph.add_edge("pull_transaction", "check_sufficiency")
        graph.add_edge("check_sufficiency", "screen")
        graph.add_conditional_edges(
            "screen", self._after_screen, {"finalize": "finalize", "retrieve": "retrieve"}
        )
        graph.add_conditional_edges(
            "retrieve",
            self._after_retrieve,
            {"broaden": "broaden", "cross_reference": "cross_reference"},
        )
        graph.add_edge("broaden", "retrieve")
        graph.add_edge("cross_reference", "assess")
        graph.add_edge("assess", "validate")
        graph.add_edge("validate", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile(checkpointer=MemorySaver())

    def assess(
        self, payload: str | dict, clarifications: dict | None = None
    ) -> Assessment:
        """Run one assessment. Re-submit with `clarifications` to complete a preliminary one."""
        thread = {"configurable": {"thread_id": uuid.uuid4().hex}}
        state = self.graph.invoke(
            {
                "raw": payload,
                "clarifications": clarifications or {},
                "trace": [],
                "errors": [],
                "attempts": 0,
                "gaps": [],
                "evidence": {},
            },
            thread,
        )
        return state["assessment"]

    @staticmethod
    def _log(state: AgentState, node: str, **detail) -> list[dict[str, Any]]:
        return state.get("trace", []) + [{"node": node, "t": round(time.time(), 3), **detail}]

    def _intake(self, state: AgentState) -> AgentState:
        raw = state["raw"]
        text = raw if isinstance(raw, str) else raw.get("description", "")
        clean, pii = redact(text)
        fields = parse_description(clean, self.book)
        if isinstance(raw, dict):
            fields.update({k: v for k, v in raw.items() if v is not None and k != "description"})
        fields.update(state.get("clarifications") or {})
        known = set(Transaction.model_fields)
        transaction = Transaction(
            description=clean, **{k: v for k, v in fields.items() if k in known}
        )
        return {
            "transaction": transaction,
            "trace": self._log(state, "intake", pii_redactions=pii, parsed=sorted(fields)),
        }

    def _pull_transaction(self, state: AgentState) -> AgentState:
        transaction = state["transaction"]
        if not transaction.transaction_id:
            return {"trace": self._log(state, "pull_transaction", skipped=True)}
        record = self.transactions.get(transaction.transaction_id)
        if record is None:
            return {
                "errors": state["errors"] + [f"transaction {transaction.transaction_id} not found"],
                "trace": self._log(state, "pull_transaction", found=False),
            }
        stored = {k: v for k, v in record.items() if k in Transaction.model_fields}
        supplied = transaction.model_dump(exclude_none=True, exclude={"description"})
        merged = {**stored, **supplied, "description": transaction.description}
        return {
            "transaction": Transaction(**merged),
            "trace": self._log(state, "pull_transaction", found=True),
        }

    def _check_sufficiency(self, state: AgentState) -> AgentState:
        """Unknown is not the same as compliant: None fields never lower a rating."""
        transaction = state["transaction"]
        questions = []
        if transaction.instrument_type is None:
            questions.append(self.book.questions["instrument_type"])
        if transaction.amount is None and transaction.exposure_pct_tier1 is None:
            questions.append(self.book.questions["amount"])
        if transaction.counterparty_kyc_verified is None and (
            transaction.instrument_type == "payment" or transaction.cross_border
        ):
            questions.append(self.book.questions["counterparty_kyc_verified"])
        if transaction.cross_border and not transaction.jurisdiction:
            questions.append(self.book.questions["jurisdiction"])
        return {"questions": questions, "trace": self._log(state, "check_sufficiency",
                                                           open_questions=len(questions))}

    def _screen(self, state: AgentState) -> AgentState:
        transaction = state["transaction"]
        triggered = self.book.evaluate(transaction)
        questions = list(state["questions"])
        for rule in triggered:
            for name in rule.get("missing", []):
                question = self.book.questions[name]
                if getattr(transaction, name) is None and question not in questions:
                    questions.append(question)
        if not triggered and not self._enough_facts(transaction):
            return {
                "triggered": [],
                "findings": [],
                "questions": questions,
                "status": AssessmentStatus.INSUFFICIENT_INFO,
                "rating": RiskRating.UNDETERMINED,
                "summary": "Not enough information to screen this transaction.",
                "trace": self._log(state, "screen", triggered=[]),
            }
        update: AgentState = {
            "triggered": triggered,
            "questions": questions,
            "trace": self._log(state, "screen", triggered=[r["id"] for r in triggered]),
        }
        if not triggered:
            # LOW means no indexed rule fired. It is not a clearance.
            update.update(
                findings=[],
                rating=RiskRating.LOW,
                status=(
                    AssessmentStatus.PRELIMINARY_NEEDS_INFO
                    if questions
                    else AssessmentStatus.COMPLETE
                ),
                summary="No screening rule fired for the facts provided.",
            )
        return update

    @staticmethod
    def _enough_facts(transaction: Transaction) -> bool:
        known = [
            transaction.amount, transaction.instrument_type, transaction.jurisdiction,
            transaction.counterparty_kyc_verified,
        ]
        return sum(value is not None for value in known) >= 3

    @staticmethod
    def _after_screen(state: AgentState) -> str:
        return "retrieve" if state.get("triggered") else "finalize"

    def _retrieve(self, state: AgentState) -> AgentState:
        evidence = dict(state.get("evidence", {}))
        errors = list(state["errors"])
        gaps: list[str] = []
        broadened = state.get("attempts", 0) > 0
        for rule in state["triggered"]:
            items: list[RetrievedChunk] = []
            for anchor in rule["anchors"]:
                found = self.tools.get_paragraph(anchor)
                if found:
                    items.append(found)
            try:
                frameworks = None if broadened else rule["frameworks"]
                hits = self.tools.query_regulations(rule["query"], frameworks)
            except ToolError as exc:
                errors.append(str(exc))
                hits = []
            seen = {i.chunk.chunk_id for i in items}
            # Anchored rules accept hits from the anchor's own document only, so a weak
            # semantic match cannot add an unrelated regulation to the assessment.
            documents = {i.chunk.doc_id for i in items}
            items.extend(
                h for h in hits
                if h.chunk.chunk_id not in seen and (not documents or h.chunk.doc_id in documents)
            )
            evidence[rule["id"]] = items[:6]
            if not items:
                gaps.append(rule["id"])
        return {
            "evidence": evidence,
            "gaps": gaps,
            "errors": errors,
            "trace": self._log(
                state,
                "retrieve",
                broadened=broadened,
                evidence={k: [i.chunk.para_ref for i in v] for k, v in evidence.items()},
            ),
        }

    def _after_retrieve(self, state: AgentState) -> str:
        retries = self.app.settings.agent.retrieval_retries
        return "broaden" if state["gaps"] and state["attempts"] < retries else "cross_reference"

    def _broaden(self, state: AgentState) -> AgentState:
        return {
            "attempts": state["attempts"] + 1,
            "trace": self._log(state, "broaden", gaps=state["gaps"]),
        }

    def _cross_reference(self, state: AgentState) -> AgentState:
        """Find transactions that trigger several frameworks or both AML and exposure duties."""
        frameworks = sorted({r["framework"] for r in state["triggered"]})
        notes = []
        if len(frameworks) > 1:
            notes.append(
                f"Obligations arise under {', '.join(frameworks)}. Each applies independently; "
                "meeting one does not discharge another."
            )
        ids = {r["id"] for r in state["triggered"]}
        if {"AML-001", "AML-002"} <= ids:
            notes.append("Missing KYC and a call-for-action jurisdiction compound each other.")
        if ids & {"LEX-001", "LEX-003"} and "LEX-004" in ids:
            notes.append(
                "Basel sets the 25% limit, while the intra-group treatment is set nationally. "
                "Test both, and apply the stricter result."
            )
        if state["transaction"].jurisdiction is None and state["transaction"].cross_border:
            notes.append("EU and US obligations cannot be scoped until the jurisdiction is known.")
        return {"cross_notes": notes, "trace": self._log(state, "cross_reference", notes=len(notes))}

    def _assess(self, state: AgentState) -> AgentState:
        findings = []
        for rule in state["triggered"]:
            items = state["evidence"].get(rule["id"], [])
            findings.append(
                Finding(
                    rule_id=rule["id"],
                    title=rule["title"],
                    framework=rule["framework"],
                    severity=RiskRating(rule["severity"]),
                    rationale=rule["rationale"],
                    required_actions=rule["actions"],
                    citations=[
                        self.tools.cite(item, position)
                        for position, item in enumerate(items[:4], start=1)
                    ],
                    evidence_found=bool(items),
                )
            )
        findings.sort(key=lambda f: RISK_ORDER[f.severity], reverse=True)
        rating = findings[0].severity if findings else RiskRating.LOW
        summary = self._narrative(state, findings, rating)
        return {
            "findings": findings,
            "rating": rating,
            "summary": summary,
            "trace": self._log(state, "assess", rating=rating.value),
        }

    def _narrative(self, state, findings, rating) -> str:
        titles = "; ".join(f.title for f in findings[:3]) or "no rule triggered"
        template = f"Risk rating {rating.value}: {titles}."
        sources = [i for f in findings for i in state["evidence"].get(f.rule_id, [])[:1]]
        if not sources or self.app.settings.llm.generation_chain[0] == "extractive":
            return template
        listed = "\n".join(
            f"[S{n}] ({s.chunk.para_ref}) {s.text}" for n, s in enumerate(sources, start=1)
        )
        facts = "\n".join(f"- {f.severity.value}: {f.title}. {f.rationale}" for f in findings)
        request = GenerationRequest(
            system=NARRATIVE_SYSTEM,
            prompt=f"TRANSACTION: {state['transaction'].description}\n"
                   f"RATING: {rating.value}\nFINDINGS:\n{facts}\nSOURCES:\n{listed}\nSUMMARY:",
            question=state["transaction"].description,
            sources=sources,
        )
        try:
            result = self.app.router.generate(
                "generation", request, {s.chunk.data_residency for s in sources}
            )
        except AllProvidersFailed as exc:
            log.warning("Narrative fell back to template: %s", exc)
            return template
        if result.provider == "extractive" or not validate_answer(result.text, sources).passed:
            return template
        return f"Risk rating {rating.value}. {result.text}"

    def _validate(self, state: AgentState) -> AgentState:
        """A finding without a citation cannot stand on its own, so it goes to a human."""
        uncited = [f.rule_id for f in state["findings"] if not f.citations]
        errors = list(state["errors"])
        if uncited:
            errors.append(f"no evidence retrieved for {uncited}")
        if errors:
            status = AssessmentStatus.HUMAN_REVIEW
        elif state["questions"]:
            status = AssessmentStatus.PRELIMINARY_NEEDS_INFO
        else:
            status = AssessmentStatus.COMPLETE
        return {
            "status": status,
            "errors": errors,
            "trace": self._log(state, "validate", status=status.value, uncited=uncited),
        }

    def _finalize(self, state: AgentState) -> AgentState:
        findings = state.get("findings", [])
        actions, citations = [], []
        seen: set[str] = set()
        for finding in findings:
            actions.extend(a for a in finding.required_actions if a not in actions)
            for citation in finding.citations:
                key = f"{citation.doc_id}|{citation.para_ref}"
                if key not in seen:
                    seen.add(key)
                    citations.append(citation)
        status = state.get("status", AssessmentStatus.INSUFFICIENT_INFO)
        rating = state.get("rating", RiskRating.UNDETERMINED)
        trace = self._log(state, "finalize")
        audit_id = self.app.audit.append(
            "assessment",
            {
                "rule_version": self.book.version,
                "status": status.value,
                "rating": rating.value,
                "transaction": state["transaction"].model_dump(),
                "rules": [f.rule_id for f in findings],
                "citations": [c.model_dump() for c in citations],
                "summary": state.get("summary", ""),
                "trace": trace,
            },
        )
        assessment = Assessment(
            assessment_id=audit_id,
            status=status,
            risk_rating=rating,
            summary=state.get("summary", ""),
            transaction=state["transaction"],
            applicable_regulations=self._regulations(citations),
            findings=findings,
            required_actions=actions,
            citations=citations,
            clarifying_questions=state.get("questions", []),
            cross_regulation_notes=state.get("cross_notes", []),
            errors=state.get("errors", []),
            trace=trace,
        )
        return {"assessment": assessment, "trace": trace}

    @staticmethod
    def _regulations(citations: list[Citation]) -> list[str]:
        return sorted({c.reference.split(",")[0].split(";")[0] for c in citations})
