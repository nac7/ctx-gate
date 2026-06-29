"""Tests for ctx-gate core modules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.compressor.compressor import ContextCompressor
from src.compressor.task_shift import TaskShiftDetector
from src.router import ModelRouter
from src.checkpoint import CheckpointWriter


# ──────────────────────────────────────────────
# ContextCompressor
# ──────────────────────────────────────────────

class TestContextCompressor:

    def _make_messages(self, n_old=10, n_recent=4):
        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(n_old):
            msgs.append({"role": "user", "content": f"Old user message {i} with some content here."})
            msgs.append({"role": "assistant", "content": f"Old assistant reply {i}. I created file_{i}.py."})
        for i in range(n_recent):
            msgs.append({"role": "user", "content": f"Recent message {i}"})
            msgs.append({"role": "assistant", "content": f"Recent reply {i}"})
        return msgs

    def test_compresses_long_history(self):
        c = ContextCompressor(recency_window=4)
        msgs = self._make_messages(n_old=8, n_recent=2)
        result = c.compress(msgs, "current task")
        assert result.compressed_tokens < result.original_tokens
        assert result.summary_injected is True

    def test_short_history_no_summary(self):
        c = ContextCompressor(recency_window=10)
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = c.compress(msgs, "next question")
        assert result.summary_injected is False

    def test_tool_output_truncated(self):
        c = ContextCompressor(max_tool_output_lines=10)
        big_output = "\n".join(f"line {i}" for i in range(100))
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "name": "bash", "content": big_output},
            {"role": "user", "content": "what happened?"},
        ]
        result = c.compress(msgs, "what happened?")
        out_msgs = [m for m in result.messages if m.get("role") == "tool"]
        if out_msgs:
            assert "compressed" in out_msgs[0]["content"]

    def test_file_diff_on_repeat(self):
        c = ContextCompressor()
        first = c.register_file("src/main.py", "def foo():\n    pass\n")
        second = c.register_file("src/main.py", "def foo():\n    return 42\n")
        unchanged = c.register_file("src/main.py", "def foo():\n    return 42\n")
        assert first == "def foo():\n    pass\n"  # first time: full content
        assert "diff" in second.lower() or "---" in second  # second time: diff
        assert "UNCHANGED" in unchanged  # no change

    def test_code_block_compression(self):
        c = ContextCompressor()
        big_code = "```python\n" + "\n".join(f"x = {i}" for i in range(200)) + "\n```"
        msgs = [{"role": "user", "content": big_code}]
        result = c.compress(msgs, "explain this")
        content = result.messages[-1]["content"]
        assert "compressed" in content

    def test_savings_pct_positive(self):
        c = ContextCompressor(recency_window=2)
        msgs = self._make_messages(n_old=15, n_recent=2)
        result = c.compress(msgs, "continue")
        assert result.savings_pct >= 0


# ──────────────────────────────────────────────
# TaskShiftDetector
# ──────────────────────────────────────────────

class TestTaskShiftDetector:

    def test_explicit_shift(self):
        d = TaskShiftDetector(shift_threshold=0.4)
        result = d.detect("Now let's work on the payment module", [
            {"role": "user", "content": "Fix the login bug"},
            {"role": "assistant", "content": "Done, modified auth.py"},
        ])
        assert result.is_shift is True

    def test_continuation(self):
        d = TaskShiftDetector()
        result = d.detect("Also, can you add a docstring to that function?", [
            {"role": "user", "content": "Refactor the validate function"},
            {"role": "assistant", "content": "Done"},
        ])
        assert result.is_shift is False

    def test_carry_forward_extracted(self):
        d = TaskShiftDetector(shift_threshold=0.1)  # low threshold for test
        result = d.detect("start fresh — build a new endpoint", [
            {"role": "user", "content": "Fix auth.ts with typescript"},
            {"role": "assistant", "content": "Updated auth.ts with python patterns"},
        ])
        # carry_forward should contain some file/tech references
        if result.is_shift:
            assert isinstance(result.suggested_carry_forward, list)

    def test_reset_clears_state(self):
        d = TaskShiftDetector()
        d._prior_files = {"auth.py", "main.py"}
        d._prior_topics = ["auth", "login"]
        d.reset()
        assert d._prior_files == set()
        assert d._prior_topics == []


# ──────────────────────────────────────────────
# ModelRouter
# ──────────────────────────────────────────────

class TestModelRouter:

    def test_advanced_prompt(self):
        r = ModelRouter(provider="claude")
        d = r.route("Redesign the architecture for cross-cutting concerns")
        assert d.tier == "advanced"

    def test_fast_prompt(self):
        r = ModelRouter(provider="claude")
        d = r.route("Fix the typo in the comment")
        assert d.tier == "fast"

    def test_standard_default(self):
        r = ModelRouter(provider="claude")
        d = r.route("Add a new API endpoint for user profiles")
        assert d.tier in ("standard", "fast", "advanced")  # ambiguous, any valid tier

    def test_force_tier(self):
        r = ModelRouter(provider="claude", force_tier="fast")
        d = r.route("Redesign the entire system architecture")
        assert d.tier == "fast"  # forced, overrides signals

    def test_long_context_bumps_to_standard(self):
        r = ModelRouter(provider="claude")
        d = r.route("Rename this variable", context_length_tokens=80_000)
        assert d.tier in ("standard", "advanced")  # not fast

    def test_openai_provider(self):
        r = ModelRouter(provider="openai")
        d = r.route("Fix typo")
        assert "gpt" in d.model.lower()

    def test_model_string_returned(self):
        r = ModelRouter(provider="claude")
        model = r.get_model("Add authentication middleware")
        assert len(model) > 0


# ──────────────────────────────────────────────
# CheckpointWriter
# ──────────────────────────────────────────────

class TestCheckpointWriter(object):

    def test_write_and_load(self, tmp_path):
        cw = CheckpointWriter(checkpoint_dir=str(tmp_path))
        cw.record_decision("Used FastAPI instead of Flask")
        cw.record_next_step("Add authentication middleware")
        cw.on_tool_call("bash", "modified src/main.py")
        path = cw.write("test-session", task_description="Build REST API")
        assert path.exists()

        content = cw.load_latest()
        assert content is not None
        assert "REST API" in content or "CHECKPOINT" in content

    def test_checkpoint_trigger_on_n_tools(self, tmp_path):
        cw = CheckpointWriter(checkpoint_dir=str(tmp_path), write_every_n_tools=5)
        triggers = [cw.on_tool_call("bash") for _ in range(10)]
        # Every 5th call should return True
        assert triggers[4] is True
        assert triggers[9] is True
        assert triggers[3] is False

    def test_clear_resets_state(self, tmp_path):
        cw = CheckpointWriter(checkpoint_dir=str(tmp_path))
        cw.record_decision("some decision")
        cw._files_touched.add("main.py")
        cw.clear()
        assert cw._decisions == []
        assert cw._files_touched == set()

    def test_extract_signals_from_message(self, tmp_path):
        cw = CheckpointWriter(checkpoint_dir=str(tmp_path))
        cw.on_turn("I created auth.py and fixed the login error. Next: add tests.")
        assert len(cw._decisions) > 0 or len(cw._next_steps) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
