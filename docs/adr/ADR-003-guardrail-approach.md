### ADR-003: Guardrail implementation approach

**Status:** Accepted. **Date:** November 2026.

**Context.** The worst failure for this system is an invented number or a wrong citation in a compliance answer. Regulators expect evidence, so each guardrail decision must itself be explainable. Guardrails run on every request, so they must be fast. Checks done entirely by an LLM add delay, cost, and a second model that can also be wrong.

**Decision.** Build **layered guardrails into the application, with rule-based checks first**:

- **Always run:** PII patterns, injection patterns, citation validity, numbers present in the cited source, and schema validation.
- **On failure:** retry once, naming the failed check; then fall back to quoted source text.
- **In production, add** a model-based entailment check as a second layer. It can reject an answer, but it cannot approve one that failed a rule-based check.

**Alternatives considered**

| Option | For | Against |
|---|---|---|
| **Rules first, model second** | Fast, explainable, testable, no extra GPU | Rules miss subtle errors, so a second layer is still needed |
| NeMo Guardrails | Rich flow control and its own rules language | Adds a framework and a new language for a small set of checks |
| Guardrails AI | Good validator library, strong on schemas | Regulatory checks would still be custom |
| LLM-as-judge on every answer | Catches meaning errors | Adds cost and delay; results vary between runs |
| Provider safety filters only | No build effort | Not aware of regulation; ties control to one provider |
| Review after the fact only | Simple | Errors reach users first |

**Consequences**

- Good: every rejection has an exact reason that can go in the audit log.
- Good: the guardrails were tested without any LLM, including a stub answer with an invented number, which was rejected.
- Cost: number matching can reject a correct calculated value. The fallback then quotes the source, which is less helpful but safe.
- Cost: passing every rule-based check does not prove an answer is right. A wrongly paraphrased clause with valid citations could pass. That is why the entailment check and human review stay in the design.
- Revisit if the entailment check rejects too many correct answers, beyond the level agreed with Compliance.
