"""
Token counting with graceful degradation.

Order of preference:
  1. tiktoken (exact for OpenAI tokenizers; a close proxy for others)
  2. char/4 heuristic (no dependency)

The compressor uses this so "tokens saved" is a real measurement rather than a
character-count guess. Install the extra with:  pip install "ctx-gate[tokenizer]"
"""

from __future__ import annotations

# tiktoken's default encoding for current OpenAI models; a reasonable proxy for
# Anthropic too (token boundaries are close enough for savings accounting).
_DEFAULT_ENCODING = "cl100k_base"

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


class TokenCounter:
    """
    Counts tokens for text and message lists.

    `accurate` is True when a real tokenizer backs the counts, False when we fall
    back to the char/4 heuristic — callers can surface this so reported savings
    are never silently overstated.
    """

    def __init__(self, encoding: str = _DEFAULT_ENCODING):
        self._enc = None
        if _TIKTOKEN_AVAILABLE:
            try:
                self._enc = tiktoken.get_encoding(encoding)
            except Exception:
                self._enc = None

    @property
    def accurate(self) -> bool:
        return self._enc is not None

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            try:
                return len(self._enc.encode(text, disallowed_special=()))
            except Exception:
                pass
        return self._heuristic(text)

    def count_messages(self, messages: list[dict]) -> int:
        """Count tokens across a message list (content + a small role overhead)."""
        total = 0
        for m in messages:
            total += self.count_text(_content_text(m))
            # ~4 tokens of structural overhead per message (role, delimiters)
            total += 4
        return total

    @staticmethod
    def _heuristic(text: str) -> int:
        return max(1, len(text) // 4)


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


# Shared default instance (constructing a tiktoken encoding isn't free).
_default_counter = TokenCounter()


def count_tokens(text: str) -> int:
    return _default_counter.count_text(text)


def count_message_tokens(messages: list[dict]) -> int:
    return _default_counter.count_messages(messages)


def is_accurate() -> bool:
    return _default_counter.accurate
