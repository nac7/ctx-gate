"""
TaskShiftDetector: Detects when the user has switched to a new task.

When a task shift is detected, the gate auto-clears conversation history
and starts a fresh context — the single most effective lever for token savings.
"""

import re
from dataclasses import dataclass


TASK_SHIFT_SIGNALS = [
    # Explicit pivots — apostrophe-tolerant
    r"(now|next)\s+let.{0,2}s\s+(work on|do|handle|tackle|build|create|fix|debug|refactor|write)",
    r"\b(can you|please)\s+(now\s+)?(work on|handle|tackle|build|create|fix|debug|refactor|write)\b",
    r"\b(new (task|issue|ticket|bug|feature|file|endpoint|component))\b",
    r"\b(forget (that|this|everything)|start over|start fresh|new chat|clear context)\b",
    r"\b(switch(ing)? to|moving on to|pivot(ing)? to|let.{0,2}s switch)\b",
    r"\bnow\s+(let.{0,2}s\s+)?(work|build|create|implement|add|fix|debug|write)\b",
]

CONTINUATION_SIGNALS = [
    r"\b(also|additionally|furthermore|and (also|now)|continuing|follow.?up)\b",
    r"\b(same (file|component|function|class)|in that (same )?(file|function))\b",
    r"\b(based on (that|the above|what you (just|said)))\b",
]

_shift_re = re.compile("|".join(TASK_SHIFT_SIGNALS), re.IGNORECASE | re.DOTALL)
_cont_re = re.compile("|".join(CONTINUATION_SIGNALS), re.IGNORECASE)


@dataclass
class ShiftResult:
    is_shift: bool
    confidence: float  # 0.0 - 1.0
    reason: str
    suggested_carry_forward: list[str]  # key facts to inject into fresh context


class TaskShiftDetector:
    """
    Determines whether the current prompt represents a new task or
    a continuation of the existing one.

    Uses a signal-scoring heuristic so no LLM call is needed for this step.
    For high-confidence use cases, an optional LLM-based classifier can override.
    """

    def __init__(self, shift_threshold: float = 0.45):
        self.shift_threshold = shift_threshold
        self._prior_files: set[str] = set()
        self._prior_topics: list[str] = []

    def detect(self, prompt: str, recent_messages: list[dict]) -> ShiftResult:
        """
        Analyze a prompt and recent history to decide if this is a new task.

        Returns a ShiftResult indicating whether to clear context.
        """
        score = 0.0
        reasons = []

        # Signal 1: Explicit shift language
        if _shift_re.search(prompt):
            score += 0.5
            reasons.append("explicit task-shift language detected")

        # Signal 2: Continuation language (negative signal)
        if _cont_re.search(prompt):
            score -= 0.3
            reasons.append("continuation language detected (staying on task)")

        # Signal 3: No file overlap with recent context
        prompt_files = self._extract_file_refs(prompt)
        if prompt_files and self._prior_files and not prompt_files & self._prior_files:
            score += 0.25
            reasons.append(f"new file domain: {prompt_files}")

        # Signal 4: Very short recent history → probably fresh anyway, but
        # don't let this override an explicit shift signal
        if len(recent_messages) <= 2 and score < 0.4:
            score -= 0.2

        # Signal 5: Topic divergence via keyword cluster shift
        prompt_keywords = self._extract_keywords(prompt)
        if self._prior_topics and not any(k in self._prior_topics for k in prompt_keywords):
            score += 0.2
            reasons.append("topic keyword cluster shifted")

        # Update state
        self._prior_files = prompt_files
        self._prior_topics = list(prompt_keywords)[:10]

        is_shift = score >= self.shift_threshold
        carry = self._extract_carry_forward(recent_messages) if is_shift else []

        return ShiftResult(
            is_shift=is_shift,
            confidence=min(max(score, 0.0), 1.0),
            reason="; ".join(reasons) if reasons else "continuation inferred",
            suggested_carry_forward=carry,
        )

    def reset(self):
        """Manually reset state (e.g., after a /clear command)."""
        self._prior_files = set()
        self._prior_topics = []

    # ------------------------------------------------------------------

    def _extract_file_refs(self, text: str) -> set[str]:
        """Find file path references in text."""
        # Matches things like src/auth.ts, ./config.py, CLAUDE.md
        return set(re.findall(r"[\w./\-]+\.\w{1,6}", text))

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords (nouns/verbs, skip stop words)."""
        stop = {"the", "a", "an", "is", "to", "of", "and", "in", "it", "you",
                "that", "this", "with", "for", "on", "at", "by", "can", "i"}
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        return {w for w in words if w not in stop}

    def _extract_carry_forward(self, messages: list[dict]) -> list[str]:
        """
        Pull key facts from recent messages to inject into the next fresh context.
        E.g., project language, framework, key constraints.
        """
        carry = []
        for msg in messages[-4:]:
            content = ""
            if isinstance(msg.get("content"), str):
                content = msg["content"]
            elif isinstance(msg.get("content"), list):
                content = " ".join(
                    b.get("text", "") for b in msg["content"] if isinstance(b, dict)
                )
            # Extract file and tech mentions to carry forward
            files = re.findall(r"[\w./\-]+\.\w{1,6}", content)
            techs = re.findall(r"\b(python|typescript|javascript|rust|go|java|react|"
                               r"django|fastapi|next\.?js|postgres|redis|docker)\b",
                               content, re.IGNORECASE)
            carry.extend(files[:3])
            carry.extend(techs[:3])
        return list(dict.fromkeys(carry))[:8]  # deduplicated, max 8 items
