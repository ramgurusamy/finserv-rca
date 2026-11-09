### ADR-002: Build versus buy for the orchestration layer

**Status:** Accepted. **Date:** November 2026.

**Context.** The orchestration layer connects rules, retrieval, models and the audit log. Regulators expect decisions to be repeatable. The client has warned against repackaged framework tutorials and against depending on one LLM provider. The options range from a managed agent platform, to a framework, to fully custom code.

**Decision.** A mix of both:

- **Use** open-source **LangGraph** only to run the graph, hold state and save checkpoints.
- **Build** the parts that carry the compliance value: rule engine, retriever, LLM gateway, guardrails and audit log. These have plain Python interfaces, and no framework types cross them.

**Alternatives considered**

| Option | For | Against |
|---|---|---|
| **LangGraph plus own services** | Clear graph, replayable trace, few dependencies | Team must learn the graph model; library upgrades need care |
| LangChain agents throughout | Quick start, many integrations | Hides the logic an auditor needs to understand |
| CrewAI | Quick multi-agent setup | Conversation-based flow is hard to make repeatable |
| Haystack | Strong RAG pipelines | Less suited to stateful flows with human checkpoints |
| Vertex AI Agent Builder | Managed, fast to deploy | Proprietary; ties orchestration to one vendor |
| Fully custom code | No dependency | Must rebuild checkpoints, retries and tracing |

**Consequences**

- Good: LangGraph can be replaced without rewriting compliance logic, because each step calls an ordinary service.
- Good: the path is fixed and every step is traced, so any decision can be replayed.
- Cost: the agent is less flexible than a free-running one. A new workflow needs a graph change and a review. This is accepted on purpose.
- Cost: the services we build need maintenance and tests. The prototype's test suite is the start of that.
- Revisit if managed platforms offer runs that are region-pinned, exportable and replayable enough to satisfy audit.
