"""
Persistent, thread-safe proxy stats.

Aggregate counters (requests proxied, tokens saved, task shifts) survive restarts
by being written to a small JSON file. Writes are atomic (temp file + replace) so
a crash mid-write can't corrupt the stored stats; failures to read/write are
swallowed — stats must never take down the proxy.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

_FIELDS = ("requests", "tokens_saved", "shifts_detected")


class StatsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data = {k: 0 for k in _FIELDS}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text())
                for k in _FIELDS:
                    self._data[k] = int(loaded.get(k, 0))
        except Exception:
            pass  # corrupt/unreadable -> start from zero

    def record_request(self, tokens_saved: int = 0, shift: bool = False) -> None:
        """Record one proxied request and persist the updated totals."""
        with self._lock:
            self._data["requests"] += 1
            self._data["tokens_saved"] += max(0, int(tokens_saved))
            if shift:
                self._data["shifts_detected"] += 1
            self._save()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._data))
            tmp.replace(self.path)  # atomic on the same filesystem
        except Exception:
            pass
