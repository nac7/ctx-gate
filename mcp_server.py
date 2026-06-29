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
import asyncio
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


async def main():
    """Simple MCP server over stdin/stdout (JSON-RPC 2.0)."""
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)
    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        lambda: asyncio.BaseProtocol(), sys.stdout.buffer
    )

    def send(obj):
        line = json.dumps(obj) + "\n"
        sys.stdout.buffer.write(line.encode())
        sys.stdout.buffer.flush()

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            req = json.loads(line.decode())
        except json.JSONDecodeError:
            continue

        method = req.get("method", "")
        req_id = req.get("id")

        if method == "initialize":
            send({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ctx-gate", "version": "0.1.0"},
                },
            })

        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_SCHEMAS}})

        elif method == "tools/call":
            tool_name = req.get("params", {}).get("name", "")
            tool_params = req.get("params", {}).get("arguments", {})
            result = handle_tool(tool_name, tool_params)
            send({
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    "isError": "error" in result,
                },
            })

        elif method == "notifications/initialized":
            pass  # no response needed

        else:
            send({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


if __name__ == "__main__":
    asyncio.run(main())
