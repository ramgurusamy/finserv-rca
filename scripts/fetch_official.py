"""Download official regulatory documents listed in data/sources.yaml, with provenance."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "official"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; finserv-rca-prototype/0.1)"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = yaml.safe_load((ROOT / "data" / "sources.yaml").read_text())
    provenance, missing = [], []
    for entry in manifest["official"]:
        target = OUT / entry["file"]
        if not target.exists():
            try:
                response = requests.get(entry["url"], headers=HEADERS, timeout=60)
                response.raise_for_status()
                if target.suffix == ".pdf" and not response.content.startswith(b"%PDF"):
                    raise ValueError("response is not a PDF")
                target.write_bytes(response.content)
            except (requests.RequestException, ValueError) as exc:
                print(f"MISSING {entry['file']}: {exc}\n  Save it by hand from: {entry['url']}")
                missing.append(entry["file"])
                continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        provenance.append(
            {
                "file": entry["file"],
                "url": entry["url"],
                "sha256": digest,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        print(f"OK {entry['file']} {digest[:12]}")
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"{len(provenance)} ready, {len(missing)} missing (digests stay in use for those).")


if __name__ == "__main__":
    main()
