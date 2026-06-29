"""Unit tests for the Anthropic -> OpenAI SSE translator (pure, no network)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.proxy.sse import (
    AnthropicToOpenAITranslator,
    parse_sse_data_line,
    format_sse,
    new_chunk_id,
)


def _run(events):
    t = AnthropicToOpenAITranslator()
    chunks = []
    for e in events:
        chunks.extend(t.feed(e))
    chunks.extend(t.finish())
    return chunks


class TestTextStream:

    def test_reconstructs_text_and_role(self):
        chunks = _run([
            {"type": "message_start", "message": {"model": "claude-sonnet-4-6"}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "Hel"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "lo"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ])
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert text == "Hello"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)

    def test_model_propagated_from_message_start(self):
        chunks = _run([{"type": "message_start", "message": {"model": "claude-opus-4-8"}}])
        assert chunks[0]["model"] == "claude-opus-4-8"

    def test_empty_text_deltas_emit_nothing(self):
        t = AnthropicToOpenAITranslator()
        out = t.feed({"type": "content_block_delta", "index": 0,
                      "delta": {"type": "text_delta", "text": ""}})
        assert out == []


class TestStopReasonMapping:

    def test_max_tokens_maps_to_length(self):
        chunks = _run([{"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}])
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"

    def test_tool_use_maps_to_tool_calls(self):
        chunks = _run([{"type": "message_delta", "delta": {"stop_reason": "tool_use"}}])
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_default_is_stop(self):
        chunks = _run([{"type": "message_start", "message": {}}])
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


class TestToolUseStream:

    def test_tool_call_name_and_arguments(self):
        chunks = _run([
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "input_json_delta", "partial_json": '{"city":'}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "input_json_delta", "partial_json": '"NYC"}'}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        ])
        tool_chunks = [c for c in chunks
                       if c["choices"][0]["delta"].get("tool_calls")]
        first = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert first["function"]["name"] == "get_weather"
        assert first["index"] == 0
        args = "".join(
            c["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments", "")
            for c in tool_chunks
        )
        assert args == '{"city":"NYC"}'


class TestHelpers:

    def test_parse_sse_data_line(self):
        assert parse_sse_data_line('data: {"type":"ping"}') == {"type": "ping"}
        assert parse_sse_data_line("event: message_start") is None
        assert parse_sse_data_line("data: [DONE]") is None
        assert parse_sse_data_line("") is None
        assert parse_sse_data_line("data: {bad json") is None

    def test_format_sse_frame(self):
        frame = format_sse({"a": 1})
        assert frame.startswith(b"data: ")
        assert frame.endswith(b"\n\n")

    def test_chunk_id_prefix(self):
        assert new_chunk_id().startswith("chatcmpl-")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
