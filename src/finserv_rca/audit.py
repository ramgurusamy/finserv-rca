"""Tamper-evident audit log.

Approach: each record embeds the hash of the previous one, so any edit breaks the
chain. Production ships these records to WORM storage (Cloud Storage bucket lock).
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLog:
    GENESIS = "0" * 64

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return self.GENESIS
        with self.path.open() as handle:
            *_, last = handle
        return json.loads(last)["hash"]

    @staticmethod
    def _digest(record: dict[str, Any]) -> str:
        body = {k: v for k, v in record.items() if k != "hash"}
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()

    def append(self, event: str, payload: dict[str, Any]) -> str:
        record = {
            "audit_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
            "prev_hash": self._last_hash(),
        }
        record["hash"] = self._digest(record)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return record["audit_id"]

    def verify(self) -> tuple[bool, int, str]:
        """Return (ok, records_checked, detail)."""
        if not self.path.exists():
            return True, 0, "no audit log yet"
        previous = self.GENESIS
        count = 0
        with self.path.open() as handle:
            for count, line in enumerate(handle, start=1):
                record = json.loads(line)
                if record["prev_hash"] != previous or record["hash"] != self._digest(record):
                    return False, count, f"chain broken at record {count}"
                previous = record["hash"]
        return True, count, "chain intact"
