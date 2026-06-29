"""
ctx-gate Proxy Server

An OpenAI-compatible proxy that sits between your IDE/tool and any LLM.
Intercepts every request, applies context compression + model routing,
then forwards to the real LLM.

Usage:
  ctx-gate serve --provider=claude --port=8080

Then point your IDE/tool to:
  http://localhost:8080/v1/chat/completions

Works with Claude Code, Cursor, Copilot Chat, Continue.dev, etc.
"""

import json
import time
import uuid
import os
import sys
import asyncio
from pathlib import Path

try:
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn
except ImportError:
    print("Run: pip install fastapi uvicorn httpx")
    sys.exit(1)

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.router import ModelRouter
from src.proxy.sessions import SessionRegistry
from src.proxy.stats import StatsStore
from src.proxy.sse import (
    AnthropicToOpenAITranslator,
    parse_sse_data_line,
    format_sse,
    DONE,
)


# ------------------------------------------------------------------
# Provider base URLs
# ------------------------------------------------------------------
PROVIDER_URLS = {
    "claude":  "https://api.anthropic.com/v1/messages",
    "openai":  "https://api.openai.com/v1/chat/completions",
    "gemini":  "https://generativelanguage.googleapis.com/v1beta/chat/completions",
    "ollama":  "http://localhost:11434/v1/chat/completions",
}

PROVIDER_API_KEY_ENV = {
    "claude":  "ANTHROPIC_API_KEY",
    "openai":  "OPENAI_API_KEY",
    "gemini":  "GOOGLE_API_KEY",
    "ollama":  None,
}


def create_app(
    provider: str = "claude",
    force_tier: str | None = None,
    recency_window: int = 6,
    checkpoint_dir: str = ".ctx-gate",
    verbose: bool = False,
    rag: bool = False,
    project_root: str = ".",
    token_budget: int | None = None,
    llm_summary: bool = False,
    max_sessions: int = 128,
    max_retries: int = 2,
    session_header: str = "x-ctx-gate-session",
) -> FastAPI:
    app = FastAPI(title="ctx-gate", version="0.1.0")

    # Optional LLM-backed rolling summary (fast tier). Falls back to extractive
    # inside the compressor if a request to it ever fails.
    summarizer_fn = None
    if llm_summary:
        try:
            from src.compressor.summarizer import make_llm_summarizer
            summarizer_fn = make_llm_summarizer(provider=provider)
            print(f"[ctx-gate] LLM rolling summary enabled (fast tier, provider={provider})")
        except Exception as e:
            print(f"[ctx-gate] LLM summary disabled: {e}")
            summarizer_fn = None

    # Per-session isolation: each client conversation gets its own compressor /
    # task-shift detector / checkpoint state, keyed by a request header.
    sessions = SessionRegistry(
        recency_window=recency_window,
        token_budget=token_budget,
        summarizer_fn=summarizer_fn,
        checkpoint_dir=checkpoint_dir,
        max_sessions=max_sessions,
    )
    router = ModelRouter(provider=provider, force_tier=force_tier)

    # Optional RAG retrieval. Falls back to a TF-IDF / in-memory store when the
    # `rag` extras aren't installed, so --rag still runs (at lower quality).
    indexer = None
    if rag:
        try:
            from src.rag.indexer import CodebaseIndexer
            indexer = CodebaseIndexer(project_root)
            idx_stats = indexer.index()
            print(f"[ctx-gate] RAG indexed {idx_stats.get('chunks_indexed', 0)} chunks "
                  f"({idx_stats.get('files_indexed', 0)} files) from {project_root}")
        except Exception as e:
            print(f"[ctx-gate] RAG disabled (index failed): {e}")
            indexer = None

    instance_id = str(uuid.uuid4())[:8]
    stats = StatsStore(str(Path(checkpoint_dir) / "stats.json"))

    base_url = PROVIDER_URLS.get(provider, PROVIDER_URLS["openai"])
    api_key_env = PROVIDER_API_KEY_ENV.get(provider)
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    # ------------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "provider": provider,
            "instance_id": instance_id,
            "active_sessions": len(sessions),
            "stats": stats.snapshot(),
        }

    @app.get("/stats")
    async def get_stats():
        snap = stats.snapshot()
        return {
            "instance_id": instance_id,
            "active_sessions": len(sessions),
            "requests_proxied": snap["requests"],
            "tokens_saved_estimate": snap["tokens_saved"],
            "task_shifts_detected": snap["shifts_detected"],
        }

    def _current_prompt(messages: list[dict]) -> str:
        """Extract the latest user message as plain text (handles block content)."""
        user_msgs = [m for m in messages if m.get("role") == "user"]
        current = user_msgs[-1]["content"] if user_msgs else ""
        if isinstance(current, list):
            current = " ".join(
                b.get("text", "") for b in current if isinstance(b, dict)
            )
        return current

    def _run_pipeline(session, messages: list[dict], current_prompt: str):
        """
        Shared request pipeline: task-shift clear → checkpoint inject →
        compress → route. Used by both the OpenAI and Anthropic endpoints.

        Operates on the given session's isolated state and records aggregate
        stats once. Returns (compression_result, routing_decision); the
        compressed, system-normalized message list is on `result.messages`.
        """
        compressor = session.compressor
        shift_detector = session.shift_detector
        checkpoint = session.checkpoint
        incoming = list(messages)  # snapshot before any task-shift trimming

        # ── Detect task shift ──
        shift = shift_detector.detect(current_prompt, messages)
        if shift.is_shift:
            if verbose:
                print(f"[ctx-gate] Task shift detected (confidence={shift.confidence:.2f}): {shift.reason}")
            carry_note = ""
            if shift.suggested_carry_forward:
                carry_note = (
                    f"\n[From previous task — carried forward: "
                    f"{', '.join(shift.suggested_carry_forward)}]"
                )
            system_msgs = [m for m in messages if m["role"] == "system"]
            if system_msgs and carry_note:
                system_msgs[0]["content"] += carry_note
            messages = system_msgs + [messages[-1]]
            checkpoint.clear()

        # ── Inject saved checkpoint on a fresh session ──
        if len(messages) <= 2:
            saved = checkpoint.load_latest()
            if saved:
                system_msgs = [m for m in messages if m["role"] == "system"]
                if system_msgs:
                    system_msgs[0]["content"] = saved + "\n\n" + system_msgs[0]["content"]
                else:
                    messages.insert(0, {"role": "system", "content": saved})

        # ── Compress context ──
        result = compressor.compress(messages, current_prompt)
        tokens_saved = max(0, result.original_tokens - result.compressed_tokens)
        if verbose:
            print(
                f"[ctx-gate] Compressed {result.original_tokens}->{result.compressed_tokens} tokens "
                f"({result.savings_pct}% saved, summary={'yes' if result.summary_injected else 'no'})"
            )

        # ── Inject RAG context (only relevant code chunks) ──
        if indexer is not None:
            result.messages, rag_saved = _inject_rag_context(
                indexer, current_prompt, result.messages
            )
            tokens_saved += rag_saved
            if verbose and rag_saved:
                print(f"[ctx-gate] RAG injected relevant chunks (~{rag_saved} tokens saved vs full files)")

        # ── Record aggregate stats once (persisted) ──
        stats.record_request(tokens_saved=tokens_saved, shift=shift.is_shift)

        # ── Route to best model ──
        routing = router.route(current_prompt, result.compressed_tokens)
        if verbose:
            if router.routes_models:
                print(f"[ctx-gate] Model: {routing.model} ({routing.tier}, reason: {routing.reason})")
            else:
                print(f"[ctx-gate] Model: client's choice kept for provider '{provider}' "
                      f"(classified tier: {routing.tier})")

        # ── Periodically checkpoint session state (never fail a request over it) ──
        try:
            written = checkpoint.observe_conversation(incoming, session.id)
            if written and verbose:
                print(f"[ctx-gate] Checkpoint written: {written}")
        except Exception as e:
            if verbose:
                print(f"[ctx-gate] Checkpoint skipped: {e}")

        return result, routing

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        session = sessions.get(request.headers.get(session_header))
        current_prompt = _current_prompt(messages)
        result, routing = _run_pipeline(session, messages, current_prompt)

        # Build upstream request — routed model is authoritative for providers
        # with a known tier map; otherwise (e.g. Ollama) keep the client's model.
        upstream_body = dict(body)
        upstream_body["messages"] = result.messages
        if router.routes_models:
            upstream_body["model"] = routing.model

        # Forward to provider
        headers = {"Content-Type": "application/json"}
        if api_key and provider == "claude":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
            upstream_body = _openai_to_anthropic(upstream_body)
            return await _forward_anthropic(upstream_body, headers, stream,
                                            convert_to_openai=True)
        elif api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            return await _forward_openai(base_url, upstream_body, headers, stream)
        else:
            # No API key — passthrough mode (Ollama or pre-authed setups)
            return await _forward_openai(base_url, upstream_body, headers, stream)

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        """
        Native Anthropic Messages API endpoint.

        This is what Claude Code (and the Anthropic SDK) actually call when you
        set ANTHROPIC_BASE_URL. We process in Anthropic format and stream the
        native SSE response straight back — no lossy format round-trip.
        """
        body = await request.json()
        stream = body.get("stream", False)

        # Anthropic keeps `system` separate from the message list; normalize it
        # into a system message so the compressor/task-shift logic can see it.
        system_text = _anthropic_system_text(body.get("system"))
        conv = body.get("messages", [])
        normalized = ([{"role": "system", "content": system_text}] if system_text else []) + conv

        session = sessions.get(request.headers.get(session_header))
        current_prompt = _current_prompt(normalized)
        result, routing = _run_pipeline(session, normalized, current_prompt)

        # Split the compressed list back into Anthropic shape: all system content
        # (original + injected summary/checkpoint) folds into the `system` field.
        new_system_parts = [
            m["content"] for m in result.messages
            if m.get("role") == "system" and isinstance(m.get("content"), str)
        ]
        new_conv = [m for m in result.messages if m.get("role") != "system"]

        upstream_body = dict(body)
        upstream_body["messages"] = new_conv
        if router.routes_models:
            upstream_body["model"] = routing.model
        if new_system_parts:
            upstream_body["system"] = "\n\n".join(new_system_parts)
        elif "system" in upstream_body:
            del upstream_body["system"]

        # Forward the caller's own auth headers upstream (Claude Code sends its
        # own key/token); fall back to the server's env key if none provided.
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
        }
        for h in ("x-api-key", "authorization", "anthropic-beta"):
            if h in request.headers:
                headers[h] = request.headers[h]
        if "x-api-key" not in headers and "authorization" not in headers and api_key:
            headers["x-api-key"] = api_key

        return await _forward_anthropic(upstream_body, headers, stream)

    # ------------------------------------------------------------------
    # Provider adapters
    # ------------------------------------------------------------------

    async def _forward_openai(url: str, body: dict, headers: dict, stream: bool):
        if stream:
            async def streamer():
                try:
                    async with httpx.AsyncClient(timeout=120) as client:
                        async with client.stream("POST", url, json=body, headers=headers) as r:
                            async for chunk in r.aiter_bytes():
                                yield chunk
                except httpx.TransportError as e:
                    yield json.dumps({"error": {"type": "upstream_unavailable",
                                                "message": str(e)}}).encode()
            return StreamingResponse(streamer(), media_type="text/event-stream")

        try:
            r = await _post_with_retries(url, body, headers, attempts=max_retries + 1)
        except httpx.TransportError as e:
            return _upstream_error(e)
        return Response(content=r.content, status_code=r.status_code,
                        media_type="application/json")

    async def _forward_anthropic(body: dict, headers: dict, stream: bool,
                                 convert_to_openai: bool = False):
        """
        Forward to Anthropic's Messages API.

        Streaming: pass the native SSE byte stream straight through (Claude Code
        needs this — buffering would break the live UI and tool-use chunks).
        Non-streaming via /v1/chat/completions converts to OpenAI shape; native
        /v1/messages callers get Anthropic's response verbatim.
        """
        url = "https://api.anthropic.com/v1/messages"

        if stream:
            if convert_to_openai:
                # OpenAI-format client streaming against Claude: translate the
                # Anthropic SSE into OpenAI chat.completion.chunk SSE on the fly.
                gen = _anthropic_to_openai_sse(url, body, headers)
            else:
                # Native /v1/messages caller: pass Anthropic SSE through verbatim.
                async def gen():
                    try:
                        async with httpx.AsyncClient(timeout=120) as client:
                            async with client.stream("POST", url, json=body, headers=headers) as r:
                                async for chunk in r.aiter_bytes():
                                    yield chunk
                    except httpx.TransportError as e:
                        yield json.dumps({"type": "error", "error": {
                            "type": "upstream_unavailable", "message": str(e)}}).encode()
                gen = gen()
            return StreamingResponse(gen, media_type="text/event-stream")

        try:
            r = await _post_with_retries(url, body, headers, attempts=max_retries + 1)
        except httpx.TransportError as e:
            return _upstream_error(e)
        # OpenAI-compat callers asked for chat.completion shape; convert.
        # Native /v1/messages callers want Anthropic's response untouched.
        if r.status_code == 200 and convert_to_openai:
            return JSONResponse(content=_anthropic_to_openai(r.json()))
        return Response(content=r.content, status_code=r.status_code,
                        media_type="application/json")

    def _anthropic_system_text(system) -> str:
        """Anthropic `system` may be a string or a list of text blocks."""
        if not system:
            return ""
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            return "\n\n".join(
                b.get("text", "") for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            )
        return str(system)

    def _openai_to_anthropic(body: dict) -> dict:
        """Convert OpenAI chat completion format to Anthropic messages format."""
        messages = body.get("messages", [])
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        conv = [m for m in messages if m["role"] != "system"]

        result = {
            "model": body.get("model", "claude-sonnet-4-6"),
            "max_tokens": body.get("max_tokens", 4096),
            "messages": conv,
        }
        if system:
            result["system"] = system
        if body.get("temperature") is not None:
            result["temperature"] = body["temperature"]
        # Carry through fields the upstream needs verbatim. `stream` is critical:
        # without it the request wouldn't actually stream and the SSE reader hangs.
        for field in ("stream", "top_p", "top_k", "stop_sequences"):
            if body.get(field) is not None:
                result[field] = body[field]
        return result

    def _anthropic_to_openai(body: dict) -> dict:
        """Convert Anthropic response to OpenAI format."""
        content = ""
        for block in body.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
        return {
            "id": body.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", ""),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": body.get("stop_reason", "stop"),
            }],
            "usage": {
                "prompt_tokens": body.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": body.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    body.get("usage", {}).get("input_tokens", 0) +
                    body.get("usage", {}).get("output_tokens", 0)
                ),
            },
        }

    return app


def run_server(
    provider: str = "claude",
    port: int = 8080,
    host: str = "127.0.0.1",
    force_tier: str | None = None,
    recency_window: int = 6,
    verbose: bool = False,
    rag: bool = False,
    project_root: str = ".",
    token_budget: int | None = None,
    llm_summary: bool = False,
    max_sessions: int = 128,
    max_retries: int = 2,
):
    app = create_app(
        provider=provider,
        force_tier=force_tier,
        recency_window=recency_window,
        verbose=verbose,
        rag=rag,
        project_root=project_root,
        token_budget=token_budget,
        llm_summary=llm_summary,
        max_sessions=max_sessions,
        max_retries=max_retries,
    )
    print(f"ctx-gate proxy -> {PROVIDER_URLS.get(provider, provider)}")
    print(f"Listening on http://{host}:{port}/v1")
    print(f"Endpoints: /v1/chat/completions (OpenAI), /v1/messages (Anthropic-native)")
    print(f"Provider: {provider} | Recency window: {recency_window} turns | RAG: {'on' if rag else 'off'}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


# Upstream statuses worth retrying (rate limit + transient server errors).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _upstream_error(exc: Exception, status: int = 502) -> "JSONResponse":
    """A clean error response when the upstream provider can't be reached."""
    return JSONResponse(status_code=status, content={"error": {
        "type": "upstream_unavailable",
        "message": f"ctx-gate could not reach the upstream provider: {exc}",
    }})


async def _post_with_retries(url: str, body: dict, headers: dict, *,
                             attempts: int, timeout: float = 120):
    """
    POST with bounded retries on transient transport errors and retryable
    statuses (exponential backoff). Returns the final httpx.Response; raises
    httpx.TransportError only if every connection attempt fails.
    """
    delay = 0.5
    for i in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, json=body, headers=headers)
            if r.status_code in _RETRYABLE_STATUS and i < attempts - 1:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            return r
        except httpx.TransportError:
            if i < attempts - 1:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise


async def _anthropic_to_openai_sse(url: str, body: dict, headers: dict):
    """
    Stream Anthropic Messages SSE from `url` and yield OpenAI chat.completion.chunk
    SSE frames. Used when an OpenAI-format client streams against a Claude backend.
    """
    translator = AnthropicToOpenAITranslator(model=body.get("model", ""))
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=body, headers=headers) as r:
                if r.status_code != 200:
                    err = await r.aread()
                    yield format_sse({"error": {
                        "message": err.decode("utf-8", "replace"),
                        "code": r.status_code,
                    }})
                    yield DONE
                    return
                async for line in r.aiter_lines():
                    event = parse_sse_data_line(line)
                    if event is None:
                        continue
                    for chunk in translator.feed(event):
                        yield format_sse(chunk)
    except httpx.TransportError as e:
        # Connection dropped (typically before streaming began) — emit a clean
        # error frame instead of letting the exception break the response.
        yield format_sse({"error": {"type": "upstream_unavailable", "message": str(e)}})
        yield DONE
        return
    for chunk in translator.finish():
        yield format_sse(chunk)
    yield DONE


def _inject_rag_context(indexer, prompt: str, messages: list[dict]) -> tuple[list[dict], int]:
    """
    Retrieve relevant code chunks and inject them into the system prompt.

    Always returns (messages, tokens_saved) so callers have one stable shape.
    """
    try:
        result = indexer.retrieve(prompt, top_k=5)
        if not result.chunks:
            return messages, 0
        ctx_block = indexer.format_for_prompt(result)
        system_msgs = [m for m in messages if m["role"] == "system"]
        if system_msgs:
            system_msgs[0]["content"] = ctx_block + "\n\n" + system_msgs[0]["content"]
        else:
            messages = [{"role": "system", "content": ctx_block}] + messages
        return messages, result.prompt_tokens_saved
    except Exception:
        return messages, 0
