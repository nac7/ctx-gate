"""
Per-session state isolation.

The compressor's file-diff snapshots, the task-shift detector's prior-file/topic
memory, and the checkpoint counters are all *conversation-specific*. A single
shared instance would bleed one client's state into another's. The registry hands
each session its own bundle, keyed by a session id (taken from a request header,
falling back to "default" so single-client use is unchanged).

Sessions are capped with LRU eviction so a long-lived proxy can't leak memory as
clients come and go.
"""

from __future__ import annotations

from collections import OrderedDict

from src.compressor.compressor import ContextCompressor
from src.compressor.task_shift import TaskShiftDetector
from src.checkpoint import CheckpointWriter


class Session:
    """One client conversation's isolated processing state."""

    def __init__(
        self,
        session_id: str,
        *,
        recency_window: int,
        token_budget: int | None,
        summarizer_fn,
        checkpoint_dir: str,
    ):
        self.id = session_id
        self.compressor = ContextCompressor(
            recency_window=recency_window,
            token_budget=token_budget,
            summarizer_fn=summarizer_fn,
        )
        self.shift_detector = TaskShiftDetector()
        self.checkpoint = CheckpointWriter(checkpoint_dir=checkpoint_dir)


class SessionRegistry:
    """Creates and caches per-session state bundles with LRU eviction."""

    def __init__(
        self,
        *,
        recency_window: int = 6,
        token_budget: int | None = None,
        summarizer_fn=None,
        checkpoint_dir: str = ".ctx-gate",
        max_sessions: int = 128,
    ):
        self._cfg = dict(
            recency_window=recency_window,
            token_budget=token_budget,
            summarizer_fn=summarizer_fn,
            checkpoint_dir=checkpoint_dir,
        )
        self.max_sessions = max_sessions
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()

    def get(self, session_id: str | None = None) -> Session:
        """Return the session for `session_id`, creating it on first use."""
        sid = session_id or "default"
        existing = self._sessions.get(sid)
        if existing is not None:
            self._sessions.move_to_end(sid)
            return existing

        session = Session(sid, **self._cfg)
        self._sessions[sid] = session
        self._sessions.move_to_end(sid)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)  # evict least-recently-used
        return session

    def __len__(self) -> int:
        return len(self._sessions)

    @property
    def active_ids(self) -> list[str]:
        return list(self._sessions.keys())
