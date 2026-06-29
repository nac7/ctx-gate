"""
Integration tests for the proxy wiring added in Phase 0:
  - Model router output is actually applied to the upstream request
  - Native /v1/messages (Anthropic) endpoint exists and round-trips system field
  - Streaming requests are passed through (not buffered/dropped)
  - File-diff injection fires through compress()
  - Checkpoints are written from conversation snapshots
  - _inject_rag_context returns a stable (messages, tokens) shape
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

import src.proxy.server as server
from src.proxy.server import create_app, _inject_rag_context
from src.router import ModelRouter
from src.compressor.compressor import ContextCompressor
from src.checkpoint import CheckpointWriter


# ── Fake upstream httpx client ────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, body: dict, status: int = 200):
        self._body = body
        self.status_code = status
        self.content = json.dumps(body).encode()

    def json(self):
        return self._body


def _anthropic_sse_text(events: list[dict]) -> str:
    """Render Anthropic events as an SSE wire string."""
    parts = []
    for e in events:
        parts.append(f"event: {e['type']}")
        parts.append("data: " + json.dumps(e))
        parts.append("")
    return "\n".join(parts) + "\n"


# A canned, realistic Anthropic streaming response: greeting text + clean stop.
_CANNED_EVENTS = [
    {"type": "message_start", "message": {"model": "claude-sonnet-4-6"}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "Hello"}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": " there"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
    {"type": "message_stop"},
]


class _FakeStream:
    def __init__(self, sse_text: str):
        self._text = sse_text
        self.status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_bytes(self):
        yield self._text.encode()

    async def aiter_lines(self):
        for line in self._text.split("\n"):
            yield line

    async def aread(self):
        return self._text.encode()


class FakeAsyncClient:
    """Captures the upstream request and returns a canned Anthropic-style reply."""
    captured: dict = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        FakeAsyncClient.captured = {"url": url, "json": json, "headers": headers}
        return _FakeResp({
            "id": "msg_1", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "model": (json or {}).get("model", ""),
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        })

    def stream(self, method, url, json=None, headers=None):
        FakeAsyncClient.captured = {"url": url, "json": json,
                                    "headers": headers, "stream": True}
        return _FakeStream(_anthropic_sse_text(_CANNED_EVENTS))


@pytest.fixture
def claude_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(server.httpx, "AsyncClient", FakeAsyncClient)
    FakeAsyncClient.captured = {}
    app = create_app(provider="claude", verbose=False)
    return TestClient(app)


# ── /v1/messages (Anthropic-native) ───────────────────────────────────────────

class TestAnthropicEndpoint:

    def test_messages_route_exists(self, claude_client):
        paths = {r.path for r in claude_client.app.routes if hasattr(r, "path")}
        assert "/v1/messages" in paths

    def test_routes_fast_model_for_trivial_prompt(self, claude_client):
        r = claude_client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6",
            "system": "You are helpful.",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Fix the typo in the comment"}],
        })
        assert r.status_code == 200
        # Router should have downgraded to the fast tier and that must reach upstream
        assert FakeAsyncClient.captured["json"]["model"] == "claude-haiku-4-5-20251001"

    def test_system_field_preserved(self, claude_client):
        claude_client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6",
            "system": "SENTINEL-SYSTEM",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "add a new endpoint"}],
        })
        body = FakeAsyncClient.captured["json"]
        assert "SENTINEL-SYSTEM" in body.get("system", "")
        # Conversation messages must not carry a synthetic system role upstream
        assert all(m["role"] != "system" for m in body["messages"])

    def test_advanced_prompt_routes_to_opus(self, claude_client):
        claude_client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6", "max_tokens": 100,
            "messages": [{"role": "user",
                          "content": "Redesign the architecture for cross-cutting concerns"}],
        })
        assert FakeAsyncClient.captured["json"]["model"] == "claude-opus-4-8"

    def test_streaming_passthrough(self, claude_client):
        r = claude_client.post("/v1/messages", json={
            "model": "claude-sonnet-4-6", "max_tokens": 100, "stream": True,
            "messages": [{"role": "user", "content": "explain this"}],
        })
        assert r.status_code == 200
        assert FakeAsyncClient.captured.get("stream") is True
        assert "message_start" in r.text


# ── /v1/chat/completions (OpenAI-compat against Claude backend) ────────────────

class TestChatCompletionsEndpoint:

    def test_converts_to_openai_shape(self, claude_client):
        r = claude_client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "what is a closure?"},
            ],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "ok"

    def test_streaming_translates_to_openai_chunks(self, claude_client):
        r = claude_client.post("/v1/chat/completions", json={
            "model": "gpt-4o", "stream": True,
            "messages": [{"role": "user", "content": "say hi"}],
        })
        assert r.status_code == 200
        body = r.text
        # OpenAI chunk shape, not raw Anthropic event names
        assert "chat.completion.chunk" in body
        assert "data: [DONE]" in body
        assert "message_start" not in body  # translated, not passed through

        # The upstream request must actually have asked to stream.
        assert FakeAsyncClient.captured["json"].get("stream") is True

        # Reconstruct the streamed assistant text from the OpenAI chunks.
        contents = []
        finish = None
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            obj = json.loads(payload)
            delta = obj["choices"][0]["delta"]
            contents.append(delta.get("content", ""))
            if obj["choices"][0]["finish_reason"]:
                finish = obj["choices"][0]["finish_reason"]
        assert "".join(contents) == "Hello there"
        assert finish == "stop"


# ── Router registry fixes ─────────────────────────────────────────────────────

class TestRouterRegistry:

    def test_advanced_is_opus_4_8(self):
        r = ModelRouter(provider="claude")
        assert r.tiers["advanced"] == "claude-opus-4-8"

    def test_tier_overrides(self):
        r = ModelRouter(provider="claude", tier_overrides={"advanced": "claude-custom"})
        assert r.route("redesign the entire architecture").model == "claude-custom"


# ── File-diff injection through compress() ────────────────────────────────────

class TestFileDiffInjection:

    def test_unchanged_then_diff(self):
        c = ContextCompressor(recency_window=10)
        v1 = "src/app.py:\n```python\ndef f():\n    return 1\n```"
        v2 = "src/app.py:\n```python\ndef f():\n    return 2\n```"
        assert "def f" in c.compress([{"role": "user", "content": v1}], "x").messages[-1]["content"]
        assert "FILE DIFF" in c.compress([{"role": "user", "content": v2}], "x").messages[-1]["content"]
        assert "UNCHANGED" in c.compress([{"role": "user", "content": v2}], "x").messages[-1]["content"]


# ── Checkpoint from conversation snapshot ─────────────────────────────────────

class TestCheckpointObservation:

    def test_writes_every_n_requests(self, tmp_path):
        cw = CheckpointWriter(checkpoint_dir=str(tmp_path), write_every_n_requests=3)
        convo = [
            {"role": "user", "content": "build the API"},
            {"role": "assistant", "content": "I created src/api.py and fixed the route."},
            {"role": "tool", "content": "wrote src/api.py"},
        ]
        assert cw.observe_conversation(convo, "s1") is None   # req 1
        assert cw.observe_conversation(convo, "s1") is None   # req 2
        path = cw.observe_conversation(convo, "s1")           # req 3 → write
        assert path is not None and Path(path).exists()
        assert "src/api.py" in cw.load_latest()


# ── RAG helper stable shape ───────────────────────────────────────────────────

class _FakeRetrieval:
    chunks = []
    prompt_tokens_saved = 0
    scores = []
    query = "q"


class _FakeIndexer:
    def retrieve(self, prompt, top_k=5):
        return _FakeRetrieval()

    def format_for_prompt(self, result):
        return "[RAG]"


class TestRagHelper:

    def test_returns_two_tuple_when_empty(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = _inject_rag_context(_FakeIndexer(), "hi", msgs)
        assert isinstance(out, tuple) and len(out) == 2
        assert out[1] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
