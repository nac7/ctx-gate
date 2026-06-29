"""
LLM-backed rolling summarizer.

Builds a `summarizer_fn(messages) -> str` that compresses old conversation turns
with a cheap, fast-tier model instead of the local extractive heuristic. Plugged
into ContextCompressor via `summarizer_fn=` (constructor) or per-call.

The compressor calls this inside a try/except and falls back to extractive
summarization on any failure, so enabling it never hard-fails a request.
"""

from __future__ import annotations

import os

from src.router import ModelRouter

_INSTRUCTION = (
    "You compress earlier turns of a coding conversation. Produce a terse bullet "
    "summary that PRESERVES: decisions made, files created/changed, key constraints "
    "and values (names, numbers, configs), and open next steps. Omit pleasantries "
    "and verbose tool output. Do not invent anything not present in the turns."
)

_PROVIDER_KEY_ENV = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "ollama": None,
}

_PROVIDER_URL = {
    "claude": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/chat/completions",
    "ollama": "http://localhost:11434/v1/chat/completions",
}


def _render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        lines.append(f"{m.get('role', 'user')}: {content}")
    return "\n".join(lines)


def make_llm_summarizer(provider: str = "claude", model: str | None = None,
                        max_tokens: int = 400):
    """
    Return a summarizer_fn using the provider's fast tier (unless `model` given).
    Requires the provider's API key in the environment (except Ollama).
    """
    import httpx

    if model is None:
        model = ModelRouter(provider=provider).tiers.get("fast")

    key_env = _PROVIDER_KEY_ENV.get(provider)
    api_key = os.environ.get(key_env, "") if key_env else ""
    if key_env and not api_key:
        raise RuntimeError(f"{key_env} not set — required for --llm-summary with {provider}.")

    url = _PROVIDER_URL.get(provider, _PROVIDER_URL["openai"])

    def summarize(messages: list[dict]) -> str:
        convo = _render(messages)
        if provider == "claude":
            body = {
                "model": model, "max_tokens": max_tokens,
                "system": _INSTRUCTION,
                "messages": [{"role": "user", "content": convo}],
            }
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            r = httpx.post(url, json=body, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            return "".join(
                b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
            ).strip()
        else:
            body = {
                "model": model, "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": _INSTRUCTION},
                    {"role": "user", "content": convo},
                ],
            }
            headers = {"content-type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            r = httpx.post(url, json=body, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()

    return summarize
