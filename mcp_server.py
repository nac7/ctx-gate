"""
ctx-gate MCP Server

Exposes ctx-gate capabilities as Claude Code tools via the Model Context Protocol.
Add to your ~/.claude/claude_desktop_config.json:

{
  "mcpServers": {
    "ctx-gate": {
      "command": "python",
      "args": ["/path/to/ctx-gate/mcp_server.py"]
    }
  }
}

Available tools:
  - compress_context   Compress conversation history before sending
  - detect_task_shift  Check if a new prompt is a task shift
  - write_checkpoint   Save current session state to disk
  - load_checkpoint    Load last checkpoint into context
  - get_stats          Show token savings for this session
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.compressor.compressor import ContextCompressor
from src.compressor.task_shift import TaskShiftDetector
from src.checkpoint import CheckpointWriter
from src.router import ModelRouter


# Shared state
_compressor = ContextCompressor()
_shift_detector = TaskShiftDetector()
_checkpoint = CheckpointWriter()
_router = ModelRouter()
_stats = {"requests": 0, "tokens_saved": 0, "shifts_detected": 0}


def handle_tool(name: str, params: dict) -> dict:
    """Dispatch tool calls from MCP."""

    if name == "compress_context":
        messages = params.get("messages", [])
        prompt = params.get("current_prompt", "")
        result = _compressor.compress(messages, prompt)
        _stats["requests"] += 1
        _stats["tokens_saved"] += max(0, result.original_tokens - result.compressed_tokens)
        return {
            "compressed_messages": result.messages,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "savings_pct": result.savings_pct,
            "summary_injected": result.summary_injected,
        }

    elif name == "detect_task_shift":
        prompt = params.get("prompt", "")
        messages = params.get("recent_messages", [])
        result = _shift_detector.detect(prompt, messages)
        if result.is_shift:
            _stats["shifts_detected"] += 1
        return {
            "is_shift": result.is_shift,
            "confidence": result.confidence,
            "reason": result.reason,
            "suggested_carry_forward": result.suggested_carry_forward,
        }

    elif name == "write_checkpoint":
        session_id = params.get("session_id", "default")
        task = params.get("task_description", "")
        path = _checkpoint.write(session_id, task_description=task)
        return {"checkpoint_path": str(path), "status": "saved"}

    elif name == "load_checkpoint":
        content = _checkpoint.load_latest()
        return {"checkpoint": content, "found": content is not None}

    elif name == "route_model":
        prompt = params.get("prompt", "")
        provider = params.get("provider", "claude")
        router = ModelRouter(provider=provider)
        decision = router.route(prompt)
        return {
            "tier": decision.tier,
            "model": decision.model,
            "reason": decision.reason,
            "confidence": decision.confidence,
        }

    elif name == "get_stats":
        return _stats

    else:
        return {"error": f"Unknown tool: {name}"}


TOOL_SCHEMAS = [
    {
        "name": "compress_context",
        "description": (
            "Compress conversation history to reduce token usage. "
            "Summarizes old turns, strips tool noise, and diffs file content. "
            "Call before sending a long conversation to the LLM."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {"type": "array", "description": "Full conversation history (OpenAI format)"},
                "current_prompt": {"type": "string", "description": "The new user message"},
            },
            "required": ["messages", "current_prompt"],
        },
    },
    {
        "name": "detect_task_shift",
        "description": (
            "Detect if the current prompt represents a new task vs a continuation. "
            "If a shift is detected, the caller should clear context and start fresh."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The new user prompt to analyze"},
                "recent_messages": {"type": "array", "description": "Last few messages for context"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "write_checkpoint",
        "description": "Save current session state to disk. Auto-called every 15 tool calls.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "task_description": {"type": "string"},
            },
        },
    },
    {
        "name": "load_checkpoint",
        "description": "Load the last saved checkpoint to restore context after a session reset.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "route_model",
        "description": "Get the recommended model tier (fast/standard/advanced) for a prompt.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "provider": {"type": "string", "enum": ["claude", "openai", "gemini", "ollama"]},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "get_stats",
        "description": "Get token savings stats for the current session.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ------------------------------------------------------------------
# JSON-RPC 2.0 dispatch (pure — no I/O, so it's unit-testable)
# ------------------------------------------------------------------

# JSON-RPC standard error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603

_SERVER_INFO = {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {}},
    "serverInfo": {"name": "ctx-gate", "version": "0.1.0"},
}


def _error(req_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _result(req_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def dispatch(req: dict) -> dict | None:
    """
    Handle one parsed JSON-RPC request and return the response dict, or None when
    no response should be sent (notifications — requests without an "id").

    Never raises: a failing tool is reported as an MCP tool error, and any other
    exception becomes a JSON-RPC internal error, so a single bad message can't
    take the server down.
    """
    if not isinstance(req, dict):
        return _error(None, INVALID_REQUEST, "Invalid Request")

    method = req.get("method")
    is_notification = "id" not in req
    req_id = req.get("id")

    if method == "initialize":
        result = dict(_SERVER_INFO)
    elif method == "tools/list":
        result = {"tools": TOOL_SCHEMAS}
    elif method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        try:
            tool_result = handle_tool(name, arguments)
        except Exception as e:  # a misbehaving tool must not crash the server
            tool_result = {"error": f"{type(e).__name__}: {e}"}
        result = {
            "content": [{"type": "text", "text": json.dumps(tool_result, indent=2)}],
            "isError": "error" in tool_result,
        }
    elif method == "ping":
        result = {}
    elif method and method.startswith("notifications/"):
        return None  # client notifications (e.g. initialized) need no reply
    else:
        return None if is_notification else _error(req_id, METHOD_NOT_FOUND,
                                                   f"Method not found: {method}")

    return None if is_notification else _result(req_id, result)


def handle_line(line: str) -> str | None:
    """Parse one input line and return the serialized response line, or None."""
    line = line.strip()
    if not line:
        return None
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        return json.dumps(_error(None, PARSE_ERROR, "Parse error"))
    try:
        resp = dispatch(req)
    except Exception as e:  # defensive: dispatch already guards, but never crash
        resp = _error(req.get("id") if isinstance(req, dict) else None,
                      INTERNAL_ERROR, f"Internal error: {e}")
    return json.dumps(resp) if resp is not None else None


# ------------------------------------------------------------------
# Synchronous stdio transport
# ------------------------------------------------------------------

def _configure_stdio() -> None:
    """
    UTF-8 in/out with no newline translation. JSON-RPC over stdio is newline
    framed, so a Windows \\r\\n rewrite would corrupt the framing; cp1252 would
    fail to encode non-ASCII tool output.
    """
    try:
        sys.stdin.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    except (AttributeError, ValueError):
        pass


def serve() -> None:
    """
    Blocking stdin/stdout JSON-RPC loop. Synchronous on purpose: the handlers are
    sync, and this avoids asyncio pipe setup that isn't supported for stdio on the
    Windows Proactor event loop. Exits cleanly on EOF or a closed output pipe.
    """
    _configure_stdio()
    while True:
        try:
            line = sys.stdin.readline()
        except (KeyboardInterrupt, EOFError):
            break
        if line == "":  # EOF — client closed the pipe
            break
        out = handle_line(line)
        if out is None:
            continue
        try:
            sys.stdout.write(out + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            break  # client went away; stop quietly


if __name__ == "__main__":
    serve()
