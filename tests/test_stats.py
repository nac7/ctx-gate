"""Tests for persistent, thread-safe proxy stats."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.proxy.stats import StatsStore


class TestStatsStore:

    def test_records_and_snapshots(self, tmp_path):
        s = StatsStore(tmp_path / "stats.json")
        s.record_request(tokens_saved=100, shift=True)
        s.record_request(tokens_saved=50, shift=False)
        snap = s.snapshot()
        assert snap == {"requests": 2, "tokens_saved": 150, "shifts_detected": 1}

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "stats.json"
        s1 = StatsStore(path)
        s1.record_request(tokens_saved=2000, shift=True)
        # A fresh store at the same path must load the prior totals (restart).
        s2 = StatsStore(path)
        assert s2.snapshot()["tokens_saved"] == 2000
        assert s2.snapshot()["requests"] == 1
        s2.record_request(tokens_saved=5)
        assert s2.snapshot()["requests"] == 2

    def test_negative_tokens_clamped(self, tmp_path):
        s = StatsStore(tmp_path / "stats.json")
        s.record_request(tokens_saved=-999)
        assert s.snapshot()["tokens_saved"] == 0

    def test_corrupt_file_starts_at_zero(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text("{ not valid json")
        s = StatsStore(path)
        assert s.snapshot() == {"requests": 0, "tokens_saved": 0, "shifts_detected": 0}

    def test_snapshot_is_a_copy(self, tmp_path):
        s = StatsStore(tmp_path / "stats.json")
        snap = s.snapshot()
        snap["requests"] = 999
        assert s.snapshot()["requests"] == 0
