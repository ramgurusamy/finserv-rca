# FinServ Regulatory Compliance Assistant

Prototype for the AI Architect assignment. It answers regulatory questions with versioned citations, screens transactions with a LangGraph agent, analyses regulatory change, produces audit-committee reports, and evaluates itself.

| Deliverable | Location |
|---|---|
| Architecture document with diagrams and ADRs | `docs/FinServ_Architecture_Document.pdf`, `docs/architecture.md`, `docs/adr/` |
| Evaluation report | `docs/evaluation/evaluation_report.md` and PDF |
| Sample outputs | `sample_outputs/` |

> **Status of the committed scores:** they come from the offline baseline, with no LLM and with paraphrased digests. Run `make full` for LLM results on official documents.

## Quick start (no LLM, about 2 minutes)

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -e ".[dev]"
rca ingest -p offline
rca ask "What are the capital adequacy requirements for Tier 1 under Basel III as amended in 2023?" -p offline
rca screen "Cross-border payment of \$2M to a non-KYC verified entity in a high-risk jurisdiction" -p offline
rca impact -p offline
rca report -p offline
rca evaluate -p offline
rca audit-verify -p offline
pytest
```

## Full run with an open LLM (laptop, CPU only)

1. Install and start [Ollama](https://ollama.com/download).
2. Run `make full`. On Windows, use WSL or Git Bash.

This installs the laptop extras (these download PyTorch and are large), pulls `granite3.3:2b` (Apache 2.0), downloads official documents where the sites allow it, re-ingests, evaluates with an LLM judge, and refreshes sample outputs. Expect tens of minutes on a CPU.

Then run `make package` to build `finserv-rca-submission.zip` next to the repo folder.

Any document that a regulator's site refuses keeps its digest. To use the official file, save it by hand under the name shown in the output, inside `data/official/`, and run `make full` again. Use another model with `RCA_OLLAMA_MODEL=mistral:7b make full`.

**The abstain threshold in `config/laptop.yaml` is not yet calibrated.** Check `refusal_accuracy` and `false_refusal_rate` in the report, and adjust `min_rerank_score` if needed.

## Architecture in the code

```
src/finserv_rca/
  ingest/        parsers, paragraph-first chunker, versioned ingestion
  retrieval/     embeddings, Qdrant, BM25, RRF, re-rankers, compression
  llm/           providers (OpenAI-compatible, extractive) and the routing gateway
  guardrails/    PII redaction, citation and number validation
  rag.py         answer flow with abstention, retry and fallback
  agent/         deterministic rules, tools, LangGraph workflow
  features/      regulatory change impact and report generation
  eval/          metrics, runner, ablation, report writer
  audit.py       hash-chained audit log
  container.py   dependency wiring
config/          offline and laptop profiles, versioned prompts
data/            digests, rules, policy register, sample transactions
eval/            golden dataset
```

Behaviour is set in `config/*.yaml` and `data/policies/rules.yaml`. Services get their dependencies only in `container.py`.

## Profiles

| | `offline` | `laptop` |
|---|---|---|
| Purpose | CI, degraded-mode path | Final results |
| LLM | None (extractive) | Ollama, with extractive fallback |
| Embedding | WordLlama, bundled in pip package | BGE-small through fastembed |
| Reranker | Feature-based | ms-marco cross-encoder |
| Documents | Digests | Official where available, digests otherwise |

## Regulatory content

`data/corpus/` holds **paragraph-referenced digests, not verbatim regulation**. They were written from official pages and carry the source reference and URL in their front matter. A few paragraph labels are digest labels, such as `ESMA-P17` or `LEX-NOTE`. Two digests have secondary sources: the bank concentration amendment comes from a TaxGuru reproduction, and the Payments Banks KYC wire paragraph from a CA Alley copy. Replace them with the official text before relying on them.

`data/demo/` is a fictional circular for the change-impact demo. `data/policies/` holds illustrative rules and a synthetic policy register. **Nothing here is legal advice, and none of it should be used for real compliance decisions.**

## Open-source note

The prototype uses open-source libraries and open-weight models. Proprietary services appear only in the architecture document, each with an open alternative named.

## Tests

`pytest` runs 52 tests covering versioning, retrieval, guardrails, router failure modes, the four sample scenarios and ambiguous inputs. A stub HTTP server covers the LLM client contract.
