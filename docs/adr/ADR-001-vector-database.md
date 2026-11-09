### ADR-001: Vector database selection

**Status:** Accepted, subject to a load test in Phase 1. **Date:** November 2026.

**Context.** The vector store must:

- keep each region's data in that region;
- filter on date, framework and user permissions on every query, because point-in-time questions are central;
- support both dense and keyword search;
- handle an initial corpus of a few million chunks.

The team has limited Kubernetes capacity, so the effort to run it matters.

**Decision.** Use self-hosted **Qdrant** on GKE in each region, with three replicas. In production, dense and sparse search run in one query. The prototype uses Qdrant's embedded mode, with BM25 kept separately.

**Alternatives considered**

| Option | For | Against |
|---|---|---|
| **Qdrant** | Strong filtering, built-in sparse vectors, same engine on a laptop and in a cluster, Apache 2.0 | One more stateful system to run |
| pgvector on AlloyDB | One database, joins with other tables, familiar | Filtered search needs careful tuning at scale; weaker built-in hybrid search |
| Weaviate | Strong built-in hybrid search | Heavier to run at this size |
| Milvus | Built for very large scale | Too many parts for a few million chunks |
| Pinecone | Fully managed | Proprietary; no self-hosted option; data location depends on the vendor |
| Vertex AI Vector Search | Managed in GCP | Proprietary, less portable, not allowed in the open-source prototype |

**Consequences**

- Good: one engine handles dense search, sparse search and date filters, with the same API from laptop to production.
- Good: each region gets its own deployment, with snapshots kept in that region.
- Cost: FinServ is responsible for upgrades, backup and recovery.
- Cost: relational data stays in AlloyDB, so a registry update and a vector update are not one transaction. A repeatable release job keeps them in step.
- Revisit if the team cannot support Qdrant, or if a pgvector benchmark on real filters performs as well, which would remove one component.
