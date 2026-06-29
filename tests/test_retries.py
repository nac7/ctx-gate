"""Tests for upstream resilience: retries + clean error responses."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import pytest

import src.proxy.server as server
from src.proxy.server import _post_with_retries, _upstream_error


class _Resp:
    def __init__(self, status):
        self.status_code = status
        self.content = b"{}"


class _SharedFakeClient:
    """One instance stands in for httpx.AsyncClient across all retry attempts.

    Each `.post` consumes the next scripted behavior: an int status code to
    return, or an Exception instance to raise.
    """
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0

    def __call__(self, *a, **k):   # acts as the AsyncClient(...) constructor
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        b = self.behaviors[self.calls]
        self.calls += 1
        if isinstance(b, Exception):
            raise b
        return _Resp(b)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def _noop(*a, **k):
        return None
    monkeypatch.setattr(server.asyncio, "sleep", _noop)


def _patch_client(monkeypatch, behaviors):
    fake = _SharedFakeClient(behaviors)
    monkeypatch.setattr(server.httpx, "AsyncClient", fake)
    return fake


class TestPostWithRetries:

    def test_succeeds_after_transient_errors(self, monkeypatch):
        fake = _patch_client(monkeypatch, [
            httpx.ConnectError("boom"), httpx.ConnectError("boom"), 200,
        ])
        r = asyncio.run(_post_with_retries("u", {}, {}, attempts=3))
        assert r.status_code == 200
        assert fake.calls == 3

    def test_retries_retryable_status(self, monkeypatch):
        fake = _patch_client(monkeypatch, [503, 200])
        r = asyncio.run(_post_with_retries("u", {}, {}, attempts=3))
        assert r.status_code == 200
        assert fake.calls == 2

    def test_gives_up_and_raises(self, monkeypatch):
        _patch_client(monkeypatch, [httpx.ConnectError("x")] * 3)
        with pytest.raises(httpx.TransportError):
            asyncio.run(_post_with_retries("u", {}, {}, attempts=3))

    def test_returns_last_status_when_attempts_exhausted(self, monkeypatch):
        # All attempts return a retryable status -> the final one is returned.
        fake = _patch_client(monkeypatch, [503, 503])
        r = asyncio.run(_post_with_retries("u", {}, {}, attempts=2))
        assert r.status_code == 503
        assert fake.calls == 2

    def test_no_retry_on_success(self, monkeypatch):
        fake = _patch_client(monkeypatch, [200, 200])
        asyncio.run(_post_with_retries("u", {}, {}, attempts=3))
        assert fake.calls == 1


class TestUpstreamError:

    def test_shape_and_status(self):
        resp = _upstream_error(RuntimeError("down"))
        assert resp.status_code == 502
        import json
        body = json.loads(bytes(resp.body))
        assert body["error"]["type"] == "upstream_unavailable"
        assert "down" in body["error"]["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
