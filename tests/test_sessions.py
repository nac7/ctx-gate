"""Tests for per-session state isolation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.proxy.sessions import SessionRegistry


def _registry(**kw):
    return SessionRegistry(recency_window=6, checkpoint_dir=str(kw.pop("dir", ".ctx-gate")), **kw)


class TestSessionRegistry:

    def test_same_id_returns_same_session(self, tmp_path):
        reg = _registry(dir=tmp_path)
        assert reg.get("alice") is reg.get("alice")

    def test_different_ids_isolated(self, tmp_path):
        reg = _registry(dir=tmp_path)
        a, b = reg.get("alice"), reg.get("bob")
        assert a is not b
        assert a.compressor is not b.compressor
        assert a.shift_detector is not b.shift_detector

    def test_none_maps_to_default(self, tmp_path):
        reg = _registry(dir=tmp_path)
        assert reg.get(None) is reg.get("default")

    def test_lru_eviction(self, tmp_path):
        reg = _registry(dir=tmp_path, max_sessions=2)
        reg.get("a"); reg.get("b"); reg.get("c")     # "a" should be evicted
        assert len(reg) == 2
        assert "a" not in reg.active_ids
        assert {"b", "c"} == set(reg.active_ids)

    def test_recent_use_avoids_eviction(self, tmp_path):
        reg = _registry(dir=tmp_path, max_sessions=2)
        reg.get("a"); reg.get("b")
        reg.get("a")          # touch "a" so "b" is now least-recently-used
        reg.get("c")          # evicts "b"
        assert "a" in reg.active_ids and "b" not in reg.active_ids

    def test_file_snapshots_are_isolated(self, tmp_path):
        # The compressor's file-diff memory must not leak across sessions.
        reg = _registry(dir=tmp_path)
        reg.get("alice").compressor.register_file("a.py", "x = 1")
        # Bob has never seen a.py -> he gets the full file, not an "unchanged" marker.
        out = reg.get("bob").compressor.register_file("a.py", "x = 1")
        assert out == "x = 1"
