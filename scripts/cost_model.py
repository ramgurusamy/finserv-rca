"""Transparent monthly cost model for 500 concurrent users and 10,000 queries a day.

Every input is a named assumption. `sourced` prices come from public price pages or
calculators (checked November 2026). `estimate` values must be confirmed with the
Google Cloud pricing calculator and a load test before any budget is committed.
"""
from __future__ import annotations

import json
from pathlib import Path

HOURS = 730
QUERIES_DAY = 10_000
BUSINESS_DAYS = 22
QUERIES_MONTH = QUERIES_DAY * BUSINESS_DAYS
TOKENS_IN, TOKENS_OUT = 1_500, 300

GPU_ON_DEMAND = 1.0413  # sourced: g2-standard-12 with 1 L4, asia-south1
GPU_CUD_1Y = 0.8718     # sourced: same, 1-year commitment
GKE_FEE = 0.10          # sourced: per cluster-hour
ALLOYDB_16_MONTH = 1891 # sourced: alloydb-16, Mumbai, third-party price tracker
APP_NODE_HOUR = 0.45    # estimate: 8-vCPU general-purpose node
QDRANT_NODE_HOUR = 0.30 # estimate: 4-vCPU high-memory node

REGIONS = {
    # name: (baseline GPUs, burst GPUs for business hours)
    "India (asia-south1)": (2, 2),
    "EU (europe-west3)": (2, 1),
    "US (us-east4)": (2, 0),
}
BURST_HOURS = 220
APP_NODES_PER_REGION = 3
QDRANT_NODES_PER_REGION = 3
PLATFORM_EXTRAS = 1500  # estimate: storage, logging, KMS, network, WORM archive, backup


def build() -> dict:
    baseline = sum(b for b, _ in REGIONS.values()) * HOURS * GPU_CUD_1Y
    burst = sum(b for _, b in REGIONS.values()) * BURST_HOURS * GPU_ON_DEMAND
    regions = len(REGIONS)
    rows = [
        ("GPU inference, baseline (6 L4, 1-year commitment)", baseline, "sourced"),
        ("GPU inference, burst (3 L4, 220 h on demand)", burst, "sourced"),
        ("GKE regional cluster fee", regions * HOURS * GKE_FEE, "sourced"),
        ("Application and retrieval nodes", regions * APP_NODES_PER_REGION * HOURS * APP_NODE_HOUR, "estimate"),
        ("Qdrant nodes", regions * QDRANT_NODES_PER_REGION * HOURS * QDRANT_NODE_HOUR, "estimate"),
        ("AlloyDB, 4 vCPU-equivalent with HA", regions * 2 * ALLOYDB_16_MONTH / 4, "derived"),
        ("Storage, logging, KMS, network, archive", PLATFORM_EXTRAS, "estimate"),
    ]
    total = sum(value for _, value, _ in rows)

    def api(price_in: float, price_out: float) -> float:
        return QUERIES_MONTH * (TOKENS_IN * price_in + TOKENS_OUT * price_out) / 1e6

    return {
        "rows": [{"item": a, "usd_month": round(b), "basis": c} for a, b, c in rows],
        "total_usd_month": round(total),
        "queries_month": QUERIES_MONTH,
        "cost_per_query_usd": round(total / QUERIES_MONTH, 3),
        "gpu_share": round((baseline + burst) / total, 3),
        "api_comparison_usd_month": {
            "Gemini 3.5 Flash-Lite, regional": round(api(0.33, 2.75)),
            "Gemini 3.5 Flash, regional": round(api(1.65, 9.90)),
            "Gemini 3.1 Pro, regional": round(api(2.20, 13.20)),
        },
        "self_hosted_gpu_usd_month": round(baseline + burst),
    }


if __name__ == "__main__":
    result = build()
    Path(__file__).parents[1].joinpath("docs/cost_model.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
