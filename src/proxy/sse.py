"""
Anthropic -> OpenAI streaming translation.

When an OpenAI-format client (Cursor, Continue.dev, an OpenAI SDK) streams against
a Claude backend, the upstream emits Anthropic Messages SSE events but the client
expects OpenAI `chat.completion.chunk` SSE. This module translates one to the other.

The translation logic is a pure, stateful object (`AnthropicToOpenAITranslator`)
so it can be unit-tested without any network. server.py wraps it with the actual
httpx stream.

Anthropic event types handled: message_start, content_block_start,
content_block_delta (text_delta + input_json_delta), message_delta, message_stop.
"""

from __future__ import annotations

import json
import time
import uuid

# Anthropic stop_reason -> OpenAI finish_reason
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}

DONE = b"data: [DONE]\n\n"


def new_chunk_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def format_sse(obj: dict) -> bytes:
    """Serialize an OpenAI chunk dict as one SSE `data:` frame."""
    return ("data: " + json.dumps(obj, separators=(",", ":")) + "\n\n").encode()


class AnthropicToOpenAITranslator:
    """
    Feed it parsed Anthropic SSE event dicts; get back OpenAI chunk dicts.

    Usage:
        t = AnthropicToOpenAITranslator(model="claude-...")
        for event in anthropic_events:
            for chunk in t.feed(event):
                emit(format_sse(chunk))
        for chunk in t.finish():
            emit(format_sse(chunk))
        emit(DONE)
    """

    def __init__(self, model: str = ""):
        self.id = new_chunk_id()
        self.created = int(time.time())
        self.model = model
        self.finish_reason = "stop"
        self._tool_index: dict[int, int] = {}  # anthropic block idx -> openai tool idx
        self._next_tool_idx = 0

    def _chunk(self, delta: dict, finish_reason: str | None) -> dict:
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def feed(self, event: dict) -> list[dict]:
        etype = event.get("type")

        if etype == "message_start":
            self.model = event.get("message", {}).get("model", self.model) or self.model
            return [self._chunk({"role": "assistant"}, None)]

        if etype == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                oai_idx = self._next_tool_idx
                self._tool_index[event.get("index", 0)] = oai_idx
                self._next_tool_idx += 1
                return [self._chunk({"tool_calls": [{
                    "index": oai_idx,
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": ""},
                }]}, None)]
            return []

        if etype == "content_block_delta":
            delta = event.get("delta", {})
            dtype = delta.get("type")
            if dtype == "text_delta":
                text = delta.get("text", "")
                return [self._chunk({"content": text}, None)] if text else []
            if dtype == "input_json_delta":
                oai_idx = self._tool_index.get(event.get("index", 0), 0)
                return [self._chunk({"tool_calls": [{
                    "index": oai_idx,
                    "function": {"arguments": delta.get("partial_json", "")},
                }]}, None)]
            return []

        if etype == "message_delta":
            stop = event.get("delta", {}).get("stop_reason")
            if stop:
                self.finish_reason = _STOP_REASON_MAP.get(stop, "stop")
            return []

        # content_block_stop, message_stop, ping -> nothing to emit
        return []

    def finish(self) -> list[dict]:
        """Final chunk carrying the finish_reason (OpenAI closes the stream this way)."""
        return [self._chunk({}, self.finish_reason)]


def parse_sse_data_line(line: str) -> dict | None:
    """Return the parsed JSON of an SSE `data:` line, or None for other lines."""
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None
