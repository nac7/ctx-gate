"""Tests for the hardened MCP stdio server."""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mcp_server
from mcp_server import dispatch, handle_line, TOOL_SCHEMAS

ROOT = Path(__file__).parent.parent


class TestDispatch:

    def test_initialize(self):
        resp = dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert resp["id"] == 1
        assert resp["result"]["serverInfo"]["name"] == "ctx-gate"
        assert "protocolVersion" in resp["result"]

    def test_tools_list(self):
        resp = dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in resp["result"]["tools"]}
        assert names == {t["name"] for t in TOOL_SCHEMAS}
        assert "compress_context" in names

    def test_tools_call_success(self):
        resp = dispatch({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "route_model",
                       "arguments": {"prompt": "redesign the whole architecture"}},
        })
        assert resp["result"]["isError"] is False
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert payload["tier"] == "advanced"

    def test_tools_call_unknown_tool_is_error(self):
        resp = dispatch({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "does_not_exist", "arguments": {}},
        })
        assert resp["result"]["isError"] is True

    def test_tool_exception_does_not_crash(self, monkeypatch):
        def boom(name, params):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(mcp_server, "handle_tool", boom)
        resp = dispatch({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "compress_context", "arguments": {}},
        })
        assert resp["result"]["isError"] is True
        assert "kaboom" in resp["result"]["content"][0]["text"]

    def test_ping(self):
        resp = dispatch({"jsonrpc": "2.0", "id": 6, "method": "ping"})
        assert resp["result"] == {}

    def test_notification_gets_no_response(self):
        assert dispatch({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    def test_unknown_method_request_errors(self):
        resp = dispatch({"jsonrpc": "2.0", "id": 7, "method": "bogus/method"})
        assert resp["error"]["code"] == -32601

    def test_unknown_method_notification_is_silent(self):
        assert dispatch({"jsonrpc": "2.0", "method": "bogus/method"}) is None

    def test_non_dict_is_invalid_request(self):
        resp = dispatch(["not", "an", "object"])
        assert resp["error"]["code"] == -32600


class TestHandleLine:

    def test_parse_error(self):
        out = handle_line('{ this is not json')
        assert json.loads(out)["error"]["code"] == -32700

    def test_blank_line_ignored(self):
        assert handle_line("   \n") is None

    def test_notification_line_ignored(self):
        assert handle_line('{"jsonrpc":"2.0","method":"notifications/initialized"}') is None

    def test_valid_line_roundtrips(self):
        out = handle_line('{"jsonrpc":"2.0","id":9,"method":"tools/list"}')
        assert json.loads(out)["id"] == 9


class TestStdioSubprocess:
    """Prove the real stdin/stdout loop runs on this platform (the Windows fix)."""

    def test_end_to_end_over_pipes(self):
        requests = "\n".join([
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "get_stats", "arguments": {}}}),
        ]) + "\n"

        proc = subprocess.run(
            [sys.executable, "mcp_server.py"],
            input=requests, capture_output=True, text=True,
            cwd=str(ROOT), timeout=60,
        )
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        responses = [json.loads(l) for l in lines]

        # The notification gets no reply -> exactly three responses, ids 1,2,3.
        assert [r["id"] for r in responses] == [1, 2, 3]
        assert responses[0]["result"]["serverInfo"]["name"] == "ctx-gate"
        assert "tools" in responses[1]["result"]
        assert responses[2]["result"]["isError"] is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
