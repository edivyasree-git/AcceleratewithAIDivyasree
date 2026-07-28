from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.config import AUDIT_DIR


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class AuditLogger:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or new_run_id()
        self.log_path = AUDIT_DIR / f"{self.run_id}.jsonl"

    def log(self, agent: str, action: str, **kwargs) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = {
            "timestamp": timestamp,
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            **kwargs,
        }
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_logs(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with self.log_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
