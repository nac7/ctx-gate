"""Tests for Phase 2 compressor features: relevance retention, token budget,
pluggable summarizer."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compressor.compressor import ContextCompressor


def _long_convo(n_pairs=10, filler=40):
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"chore {i}: " + "filler " * filler})
        msgs.append({"role": "assistant", "content": f"did chore {i}: " + "noise " * filler})
    return msgs


class TestRelevanceRetention:

    def test_keeps_relevant_old_turn_verbatim(self):
        # Bury a fact early, pad it out of the recency window, then probe for it.
        msgs = (
            [{"role": "user", "content": "The cache TTL must be 900 seconds, non-negotiable."},
             {"role": "assistant", "content": "Understood."}]
            + [{"role": "user", "content": f"tweak thing {i}"} for i in range(12)]
        )
        c = ContextCompressor(recency_window=4, relevance_keep=2)
        out = c.compress(msgs, "what is the cache TTL again?")
        text = "\n".join(m.get("content", "") for m in out.messages)
        assert "900 seconds" in text  # relevant old turn was kept verbatim

    def test_relevance_keep_zero_disables(self):
        msgs = (
            [{"role": "assistant", "content": "The cache TTL is 900 seconds."}]
            + [{"role": "user", "content": f"unrelated chore {i}"} for i in range(12)]
        )
        c = ContextCompressor(recency_window=4, relevance_keep=0)
        out = c.compress(msgs, "what is the cache TTL?")
        text = "\n".join(m.get("content", "") for m in out.messages)
        # With no relevance retention and no signal verbs, the fact is summarized away.
        assert "900 seconds" not in text


class TestTokenBudget:

    def test_enforces_budget(self):
        msgs = _long_convo(12, filler=60)
        msgs.append({"role": "user", "content": "summary please"})
        c = ContextCompressor(recency_window=10, token_budget=300)
        out = c.compress(msgs, "summary please")
        assert out.compressed_tokens <= 300

    def test_never_drops_system_or_current_prompt(self):
        msgs = _long_convo(12, filler=60)
        msgs.append({"role": "user", "content": "FINAL QUESTION sentinel"})
        c = ContextCompressor(recency_window=10, token_budget=200)
        out = c.compress(msgs, "FINAL QUESTION sentinel")
        assert any(m["role"] == "system" for m in out.messages)
        assert out.messages[-1]["content"] == "FINAL QUESTION sentinel"

    def test_no_budget_leaves_more_tokens(self):
        msgs = _long_convo(12, filler=60)
        msgs.append({"role": "user", "content": "q"})
        budgeted = ContextCompressor(recency_window=10, token_budget=300).compress(msgs, "q")
        free = ContextCompressor(recency_window=10).compress(msgs, "q")
        assert free.compressed_tokens > budgeted.compressed_tokens


class TestPluggableSummarizer:

    def test_default_summarizer_used(self):
        seen = {"n": 0}

        def fake(messages):
            seen["n"] += 1
            return "MOCK-SUMMARY"

        c = ContextCompressor(recency_window=2, summarizer_fn=fake)
        out = c.compress(_long_convo(6), "next")
        assert seen["n"] == 1
        assert any("MOCK-SUMMARY" in (m.get("content") or "") for m in out.messages)

    def test_per_call_overrides_default(self):
        c = ContextCompressor(recency_window=2, summarizer_fn=lambda m: "DEFAULT")
        out = c.compress(_long_convo(6), "next", summarizer_fn=lambda m: "PERCALL")
        text = "\n".join(m.get("content", "") for m in out.messages)
        assert "PERCALL" in text and "DEFAULT" not in text

    def test_falls_back_to_extractive_on_failure(self):
        def boom(messages):
            raise RuntimeError("provider down")

        c = ContextCompressor(recency_window=2, summarizer_fn=boom)
        out = c.compress(_long_convo(6), "next")
        # Extractive summary block still present despite the summarizer failing.
        assert any("CONTEXT SUMMARY" in (m.get("content") or "") for m in out.messages)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
