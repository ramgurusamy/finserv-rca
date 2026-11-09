"""Generate the sample outputs committed with the repository."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from finserv_rca.container import build

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample_outputs"
SCENARIOS = {
    "tx1_cross_border_non_kyc": "Cross-border payment of $2M to a non-KYC verified entity in a high-risk jurisdiction",
    "tx2_intragroup_derivative": "Intra-group derivative trade exceeding large exposure threshold",
    "tx3_mifid_appropriateness": "Retail customer investment in a complex product without appropriateness assessment",
    "tx4_nbfc_priority_sector": "NBFC lending transaction requiring priority sector reporting",
    "tx5_ambiguous": "Please check this payment",
}
QUESTIONS = {
    "qa1_basel_tier1": "What are the capital adequacy requirements for Tier 1 under Basel III as amended in 2023?",
    "qa2_refusal": "What does the Dodd-Frank Volcker Rule say about proprietary trading?",
}


def main(profile: str) -> None:
    OUT.mkdir(exist_ok=True)
    app = build(profile)
    for name, question in QUESTIONS.items():
        (OUT / f"{name}.json").write_text(app.rag.ask(question).model_dump_json(indent=2))
    for name, text in SCENARIOS.items():
        assessment = app.agent.assess(text)
        (OUT / f"{name}.json").write_text(assessment.model_dump_json(indent=2))
    # The second run shows how a preliminary assessment is completed with answers.
    follow_up = app.agent.assess(
        SCENARIOS["tx1_cross_border_non_kyc"], clarifications={"jurisdiction": "Myanmar"}
    )
    (OUT / "tx1_after_clarification.json").write_text(follow_up.model_dump_json(indent=2))
    app.close()
    for args, target in (
        (["impact", "-p", profile], "impact_demo_circular.json"),
        (["report", "-p", profile], None),
    ):
        output = subprocess.run(["rca", *args], check=True, capture_output=True, text=True).stdout
        if target:
            (OUT / target).write_text(output)
    print(f"Sample outputs written to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "offline")
