"""Tests for token counting and its wiring into the compressor."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compressor.tokens import TokenCounter, count_tokens, count_message_tokens
from src.compressor.compressor import ContextCompressor


class TestTokenCounter:

    def test_empty_is_zero(self):
        assert TokenCounter().count_text("") == 0

    def test_longer_text_more_tokens(self):
        tc = TokenCounter()
        assert tc.count_text("a much longer piece of text here") > tc.count_text("hi")

    def test_messages_include_overhead(self):
        tc = TokenCounter()
        # Two empty messages still cost the per-message structural overhead.
        assert tc.count_messages([{"role": "user", "content": ""},
                                  {"role": "assistant", "content": ""}]) >= 8

    def test_module_helpers(self):
        assert count_tokens("hello world") > 0
        assert count_message_tokens([{"role": "user", "content": "hello world"}]) > 0

    def test_accurate_flag_is_bool(self):
        assert isinstance(TokenCounter().accurate, bool)

    def test_heuristic_fallback_path(self):
        # Force the no-tokenizer path and confirm it still returns sane counts.
        tc = TokenCounter()
        tc._enc = None
        assert tc.count_text("a" * 40) == 10  # char/4


class TestCompressorUsesCounter:

    def test_savings_positive_on_long_history(self):
        c = ContextCompressor(recency_window=2)
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(12):
            msgs.append({"role": "user", "content": f"message {i} " + "word " * 80})
            msgs.append({"role": "assistant", "content": f"reply {i} " + "text " * 80})
        result = c.compress(msgs, "continue")
        assert result.compressed_tokens < result.original_tokens
        assert result.savings_pct > 0
