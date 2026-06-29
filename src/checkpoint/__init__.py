"""
CheckpointWriter: Persists session progress to disk every N tool calls.

When compaction or a session limit drops context mid-session, the checkpoint
file is injected into the next session's system prompt so nothing is lost.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Checkpoint:
    session_id: str
    timestamp: float
    turn_count: int
    tool_call_count: int
    task_description: str
    decisions: list[str]
    files_touched: list[str]
    next_steps: list[str]
    carry_forward: dict  # arbitrary key/value context


class CheckpointWriter:
    """
    Writes session checkpoints to a `.ctx-gate/` directory in the project root.

    The checkpoint is also returned as a formatted string so it can be prepended
    to a new session's system prompt automatically.
    """

    def __init__(
        self,
        checkpoint_dir: str = ".ctx-gate",
        write_every_n_tools: int = 15,
        write_every_n_requests: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.write_every_n_tools = write_every_n_tools
        self.write_every_n_requests = write_every_n_requests
        self._tool_call_count = 0
        self._turn_count = 0
        self._request_count = 0
        self._decisions: list[str] = []
        self._files_touched: set[str] = set()
        self._next_steps: list[str] = []

    # ------------------------------------------------------------------
    # Call these from the proxy on each event
    # ------------------------------------------------------------------

    def on_tool_call(self, tool_name: str, tool_output: str = ""):
        """Called when the LLM executes a tool."""
        self._tool_call_count += 1
        # Extract file paths from tool output
        import re
        files = re.findall(r"[\w./\-]+\.\w{1,6}", tool_output)
        self._files_touched.update(files[:5])

        if self._tool_call_count % self.write_every_n_tools == 0:
            return True  # signal: write checkpoint now
        return False

    def on_turn(self, assistant_message: str):
        """Called after each assistant turn to extract signals."""
        self._turn_count += 1
        self._extract_from_message(assistant_message)

    def record_decision(self, decision: str):
        """Manually record a key decision."""
        self._decisions.append(decision)

    def record_next_step(self, step: str):
        """Manually record what needs to happen next."""
        self._next_steps.append(step)

    def observe_conversation(
        self, messages: list[dict], session_id: str, task_description: str = ""
    ) -> "Path | None":
        """
        Proxy-friendly entry point. A chat proxy can't see individual tool-call
        events, but it sees the whole conversation on every request. This derives
        checkpoint state from that snapshot (idempotently) and writes a checkpoint
        every `write_every_n_requests` requests. Returns the path written, or None.
        """
        import re
        self._request_count += 1

        # Rebuild state from the full history so re-sent turns don't double-count.
        self._decisions = []
        self._next_steps = []
        self._files_touched = set()
        turns = 0
        tool_calls = 0
        file_re = re.compile(r"[\w./\-]+\.\w{1,6}")

        for msg in messages:
            content = self._content_text(msg)
            role = msg.get("role")
            if role == "assistant":
                turns += 1
                self._extract_from_message(content)
            elif role == "tool":
                tool_calls += 1
                self._files_touched.update(file_re.findall(content)[:5])

        self._turn_count = turns
        self._tool_call_count = tool_calls

        if self._request_count % self.write_every_n_requests == 0:
            return self.write(session_id, task_description=task_description)
        return None

    @staticmethod
    def _content_text(msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return str(content)

    # ------------------------------------------------------------------

    def write(self, session_id: str, task_description: str = "", carry: dict | None = None) -> Path:
        """Write a checkpoint file and return its path."""
        cp = Checkpoint(
            session_id=session_id,
            timestamp=time.time(),
            turn_count=self._turn_count,
            tool_call_count=self._tool_call_count,
            task_description=task_description,
            decisions=self._decisions[-10:],
            files_touched=sorted(self._files_touched),
            next_steps=self._next_steps[-5:],
            carry_forward=carry or {},
        )
        path = self.checkpoint_dir / f"checkpoint-{session_id}.json"
        path.write_text(json.dumps(asdict(cp), indent=2))
        return path

    def load_latest(self) -> str | None:
        """Load the most recent checkpoint as a formatted system prompt injection."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*.json"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        if not checkpoints:
            return None

        cp = json.loads(checkpoints[0].read_text())
        return self._format_for_injection(cp)

    def clear(self):
        """Reset in-memory state (call after /clear or task shift)."""
        self._tool_call_count = 0
        self._turn_count = 0
        self._request_count = 0
        self._decisions = []
        self._files_touched = set()
        self._next_steps = []

    # ------------------------------------------------------------------

    def _extract_from_message(self, text: str):
        """Auto-extract decisions and next-steps from assistant output."""
        import re
        lines = text.splitlines()
        for line in lines:
            stripped = line.strip()
            # Detect decision-like statements
            if re.match(r"^(I (will|decided|chose|used|implemented|fixed|created|"
                        r"added|wrote|built|refactored|updated)|"
                        r"Decision:|Approach:|Note:)", stripped, re.IGNORECASE):
                self._decisions.append(stripped[:200])
            # Detect next-step hints (line-start or mid-line "Next:" clause)
            if re.search(r"(^|\.\s+)(Next[: ]|TODO[: ]|Step \d|Then |After this)",
                         stripped, re.IGNORECASE):
                self._next_steps.append(stripped[:200])

    def _format_for_injection(self, cp: dict) -> str:
        """Format checkpoint as a concise system prompt block."""
        parts = ["[RESTORED SESSION CHECKPOINT]"]
        if cp.get("task_description"):
            parts.append(f"Task: {cp['task_description']}")
        if cp.get("files_touched"):
            parts.append(f"Files touched: {', '.join(cp['files_touched'][:10])}")
        if cp.get("decisions"):
            parts.append("Key decisions:")
            for d in cp["decisions"]:
                parts.append(f"  • {d}")
        if cp.get("next_steps"):
            parts.append("Next steps:")
            for s in cp["next_steps"]:
                parts.append(f"  • {s}")
        if cp.get("carry_forward"):
            parts.append(f"Context: {json.dumps(cp['carry_forward'])}")
        parts.append(f"[Turn {cp['turn_count']}, {cp['tool_call_count']} tool calls when saved]")
        return "\n".join(parts)
