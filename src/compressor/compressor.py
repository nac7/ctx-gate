"""
ContextCompressor: Semantic compression of conversation history and context.

Strategies:
1. Rolling summary of old turns beyond a recency window
2. Diff-based file injection (only changed regions, not whole files)
3. Tool output truncation (keep signal, strip noise)
4. Relevance scoring (drop low-relevance context chunks)
"""

import re
import difflib
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .tokens import TokenCounter


@dataclass
class Message:
    role: str
    content: str
    tokens: int = 0
    turn_index: int = 0
    task_id: str = ""


@dataclass
class CompressionResult:
    messages: list[dict]
    original_tokens: int
    compressed_tokens: int
    summary_injected: bool
    savings_pct: float


class ContextCompressor:
    """
    Compresses conversation history to reduce token usage while preserving accuracy.

    Key behaviors:
    - Keeps the last `recency_window` turns verbatim (recent = most relevant)
    - Summarizes older turns into a single structured memory block
    - Strips tool output noise (long logs, stack traces) to key lines
    - Performs diff-based file injection (only send deltas, not full files)
    - Scores each context chunk for relevance to the current user prompt
    """

    def __init__(
        self,
        recency_window: int = 6,
        max_tool_output_lines: int = 40,
        summary_model: str | None = None,
        token_budget: int | None = None,
        relevance_keep: int = 2,
        summarizer_fn=None,
    ):
        self.recency_window = recency_window
        self.max_tool_output_lines = max_tool_output_lines
        self.summary_model = summary_model
        self.token_budget = token_budget
        # How many old turns (beyond the recency window) to keep verbatim when
        # they're relevant to the current prompt, instead of summarizing them away.
        self.relevance_keep = relevance_keep
        # Optional default summarizer (e.g. a fast-tier LLM). Falls back to the
        # extractive summarizer when None or when a per-call fn isn't supplied.
        self._default_summarizer = summarizer_fn
        self._counter = TokenCounter()
        self._file_snapshots: dict[str, str] = {}  # path -> last injected content

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        messages: list[dict],
        current_prompt: str,
        summarizer_fn=None,
    ) -> CompressionResult:
        """
        Main entry point. Takes raw OpenAI-format messages and returns a
        compressed version ready to send to any LLM.

        Args:
            messages:        Full conversation history (OpenAI format)
            current_prompt:  The new user message being sent
            summarizer_fn:   Optional callable(messages: list[dict]) -> str
                             for LLM-based summarization of old turns.
                             Falls back to extractive summarization if None.
        """
        original_tokens = self._estimate_tokens(messages)

        # 1. Split into old history (to compress) and recent window (keep verbatim)
        system_msgs = [m for m in messages if m["role"] == "system"]
        conv_msgs = [m for m in messages if m["role"] != "system"]

        recent = conv_msgs[-self.recency_window:]
        older = conv_msgs[: max(0, len(conv_msgs) - self.recency_window)]

        # 2. Among the old turns, keep the few most relevant to the current prompt
        #    verbatim; only the rest get summarized. This is what stops the
        #    summarizer from dropping a buried fact the user is now asking about.
        keep_relevant = self._select_relevant(older, current_prompt, self.relevance_keep)
        keep_ids = {id(m) for m in keep_relevant}
        to_summarize = [m for m in older if id(m) not in keep_ids]

        compressed = list(system_msgs)
        summary_injected = False

        # 3. Summarize the non-relevant old turns
        if to_summarize:
            summary = self._summarize(to_summarize, summarizer_fn, current_prompt)
            compressed.append({
                "role": "system",
                "content": f"[CONTEXT SUMMARY — earlier turns]\n{summary}",
            })
            summary_injected = True

        # 4. Re-insert the relevant old turns (kept in original order)
        for msg in keep_relevant:
            compressed.append(self._process_message(msg, current_prompt))

        # 5. Process recent turns: strip tool noise, compress file blocks
        for msg in recent:
            compressed.append(self._process_message(msg, current_prompt))

        # 6. Enforce a hard token budget if one is set (drop least-relevant first)
        if self.token_budget:
            compressed = self._enforce_budget(compressed, current_prompt)

        compressed_tokens = self._estimate_tokens(compressed)
        savings = max(0.0, (original_tokens - compressed_tokens) / max(original_tokens, 1)) * 100

        return CompressionResult(
            messages=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            summary_injected=summary_injected,
            savings_pct=round(savings, 1),
        )

    def register_file(self, path: str, content: str) -> str:
        """
        Register a file snapshot. Returns a diff if we've seen this file before,
        or the full content on first injection. Saves tokens on repeated file loads.
        """
        if path not in self._file_snapshots:
            self._file_snapshots[path] = content
            return content

        old = self._file_snapshots[path]
        if old == content:
            return f"[FILE UNCHANGED: {path}]"

        diff = self._make_diff(path, old, content)
        self._file_snapshots[path] = content
        return diff

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _summarize(self, messages: list[dict], summarizer_fn=None,
                   current_prompt: str = "") -> str:
        """
        Produce a compact summary of old conversation turns.

        Uses, in order: a per-call summarizer_fn, then a default summarizer
        (e.g. a fast-tier LLM) configured on the compressor, then the local
        extractive summarizer.
        """
        fn = summarizer_fn or self._default_summarizer
        if fn:
            try:
                return fn(messages)
            except Exception:
                pass  # fall back to extractive on any LLM failure
        return self._extractive_summary(messages)

    def _extractive_summary(self, messages: list[dict]) -> str:
        """
        Lightweight extractive summarization without an LLM call.
        Extracts key decisions, file touches, and task outcomes.
        """
        lines = []
        for msg in messages:
            role = msg["role"]
            content = self._get_content_text(msg)
            if not content.strip():
                continue

            if role == "user":
                # Keep first sentence of user messages (usually the task description)
                first = content.strip().split("\n")[0][:200]
                lines.append(f"• User: {first}")
            elif role == "assistant":
                # Extract key signals: file mentions, decisions, errors
                signals = self._extract_signals(content)
                if signals:
                    lines.append(f"• Assistant: {signals}")
            elif role == "tool":
                # Just note that a tool ran; drop verbose output
                tool_name = msg.get("name", "tool")
                lines.append(f"• Tool [{tool_name}]: ran (output compressed)")

        return "\n".join(lines) if lines else "No significant prior context."

    # ------------------------------------------------------------------
    # Relevance scoring & budget enforcement
    # ------------------------------------------------------------------

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
        "in", "it", "you", "we", "they", "that", "this", "with", "for", "on",
        "at", "by", "can", "i", "do", "did", "does", "what", "which", "how",
        "why", "when", "me", "my", "our", "your", "have", "has", "had", "be",
        "been", "will", "would", "should", "could", "remind", "tell", "show",
    }

    def _informative_tokens(self, text: str) -> set[str]:
        """Lowercased content words (len>=3, no stopwords) used for relevance."""
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{2,}\b", text.lower())
        return {w for w in words if w not in self._STOPWORDS}

    def _relevance_score(self, text: str, query_tokens: set[str]) -> int:
        """Number of distinct informative tokens shared with the query."""
        if not query_tokens:
            return 0
        return len(self._informative_tokens(text) & query_tokens)

    def _select_relevant(self, messages: list[dict], current_prompt: str,
                         top_n: int) -> list[dict]:
        """
        Pick up to `top_n` old messages most relevant to the current prompt,
        returned in their original order. These are kept verbatim instead of
        being summarized, so a fact the user is asking about isn't lost.
        """
        if top_n <= 0 or not messages:
            return []
        query = self._informative_tokens(current_prompt)
        if not query:
            return []
        scored = []
        for idx, msg in enumerate(messages):
            score = self._relevance_score(self._get_content_text(msg), query)
            if score > 0:
                scored.append((score, idx, msg))
        scored.sort(key=lambda t: t[0], reverse=True)
        chosen = scored[:top_n]
        chosen.sort(key=lambda t: t[1])  # restore original order
        return [m for _, _, m in chosen]

    def _enforce_budget(self, messages: list[dict], current_prompt: str) -> list[dict]:
        """
        Drop messages (least relevant first) until under `token_budget`.
        System messages and the final (current) message are never dropped.
        """
        if not self.token_budget or self._estimate_tokens(messages) <= self.token_budget:
            return messages

        query = self._informative_tokens(current_prompt)
        last_idx = len(messages) - 1

        def _protected(i: int, m: dict) -> bool:
            # Never drop the current (final) message or a real input system
            # message. The compressor's own generated summary IS droppable —
            # it's a nice-to-have, and protecting it could make the cap unreachable.
            if i == last_idx:
                return True
            if m.get("role") == "system":
                content = self._get_content_text(m)
                return not content.startswith("[CONTEXT SUMMARY")
            return False

        droppable = [i for i, m in enumerate(messages) if not _protected(i, m)]
        # Least relevant first; ties broken by larger token cost (drop bloat sooner).
        droppable.sort(
            key=lambda i: (
                self._relevance_score(self._get_content_text(messages[i]), query),
                -self._counter.count_text(self._get_content_text(messages[i])),
            )
        )

        dropped: set[int] = set()
        for i in droppable:
            kept = [m for j, m in enumerate(messages) if j not in dropped]
            if self._estimate_tokens(kept) <= self.token_budget:
                break
            dropped.add(i)
        return [m for j, m in enumerate(messages) if j not in dropped]

    def _extract_signals(self, text: str) -> str:
        """Extract decision-relevant sentences from assistant output."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        signal_patterns = [
            r"\b(created|modified|deleted|wrote|updated|fixed|refactored)\b",
            r"\b(error|warning|failed|succeeded|completed)\b",
            r"\b(decision|approach|strategy|plan|next step)\b",
            r"`[^`]+\.(py|js|ts|go|rs|java|cpp|c|rb|sh)`",  # file references
        ]
        pattern = re.compile("|".join(signal_patterns), re.IGNORECASE)
        signals = [s for s in sentences if pattern.search(s)]
        return " ".join(signals[:3])[:400]  # max 3 signal sentences, 400 chars

    def _process_message(self, msg: dict, current_prompt: str) -> dict:
        """Clean up a single message: strip tool noise, diff repeats, compress blocks."""
        content = self._get_content_text(msg)

        # Truncate long tool outputs
        if msg["role"] == "tool" or self._looks_like_tool_output(content):
            content = self._truncate_tool_output(content)

        # Replace re-sent file blocks with diffs / unchanged markers
        content = self._inject_file_diffs(content)

        # Compress code fences that exceed a threshold
        content = self._compress_code_blocks(content)

        result = dict(msg)
        if isinstance(msg.get("content"), str):
            result["content"] = content
        return result

    # A "file block" = a header line ending in a file path, immediately followed
    # by a fenced code block. This is the convention agents use when they paste a
    # file's contents (e.g. "src/main.py:" then a ``` fence), so it's where
    # repeated full-file injections show up and where diffing pays off.
    _FILE_BLOCK_RE = re.compile(
        r"(?P<hdr>[^\n]*?(?P<path>[\w./\\-]+\.\w{1,8}))[ \t]*:?[ \t]*\n"
        r"```(?P<lang>[\w+\-]*)\n(?P<code>.*?)\n```",
        re.DOTALL,
    )

    def _inject_file_diffs(self, text: str) -> str:
        """
        Detect re-sent file blocks and collapse them to a diff (if the file
        changed) or an "[UNCHANGED]" marker (if identical to last time we saw it).
        First sighting is left intact so the model still gets the full file once.
        """
        def repl(match: "re.Match") -> str:
            path = match.group("path")
            code = match.group("code")
            injected = self.register_file(path, code)
            if injected == code:
                # First time we've seen this path — keep the original block.
                return match.group(0)
            # Repeat: replace the full block with a compact diff / marker.
            return f"{match.group('hdr')}\n{injected}"

        return self._FILE_BLOCK_RE.sub(repl, text)

    def _truncate_tool_output(self, text: str) -> str:
        """Keep first and last N lines of verbose tool output."""
        lines = text.splitlines()
        if len(lines) <= self.max_tool_output_lines:
            return text
        keep = self.max_tool_output_lines // 2
        head = lines[:keep]
        tail = lines[-keep:]
        dropped = len(lines) - self.max_tool_output_lines
        return "\n".join(head + [f"... [{dropped} lines compressed] ..."] + tail)

    def _compress_code_blocks(self, text: str, max_lines: int = 80) -> str:
        """Truncate very large code fences (e.g., pasted full files)."""
        def truncate_block(match):
            lang = match.group(1)
            code = match.group(2)
            lines = code.splitlines()
            if len(lines) <= max_lines:
                return match.group(0)
            keep = max_lines // 2
            dropped = len(lines) - max_lines
            compressed = "\n".join(
                lines[:keep]
                + [f"# ... [{dropped} lines compressed — use file reference instead] ..."]
                + lines[-keep:]
            )
            return f"```{lang}\n{compressed}\n```"

        return re.sub(r"```(\w*)\n(.*?)```", truncate_block, text, flags=re.DOTALL)

    def _make_diff(self, path: str, old: str, new: str) -> str:
        """Generate a unified diff between two file versions."""
        diff_lines = list(difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{path} (previous)",
            tofile=f"{path} (current)",
            n=3,
        ))
        if not diff_lines:
            return f"[FILE UNCHANGED: {path}]"
        diff_text = "".join(diff_lines)
        return f"[FILE DIFF: {path}]\n```diff\n{diff_text}\n```"

    def _looks_like_tool_output(self, text: str) -> bool:
        """Heuristic: long text with many lines is likely tool output."""
        return text.count("\n") > 30 and len(text) > 2000

    def _get_content_text(self, msg: dict) -> str:
        """Extract plain text from a message, handling list content blocks."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
        return str(content)

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """
        Count tokens via tiktoken when available, else a char/4 heuristic.
        See `TokenCounter.accurate` to tell which backed a given count.
        """
        return self._counter.count_messages(messages)
