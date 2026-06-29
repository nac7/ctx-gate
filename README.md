# ctx-gate

**LLM-agnostic context optimization proxy.** Sits between your IDE/tool and any LLM, automatically reducing session token consumption without losing the facts your next prompt depends on — and without changes to your workflow.

How much it saves depends entirely on the session: short Q&A compresses very little, while long sessions with verbose tool output compress heavily. ctx-gate doesn't ask you to trust that number — it ships a [faithfulness harness](#faithfulness--evaluation) that measures savings **and** information retention on every change.

Speaks two APIs:
- **`POST /v1/messages`** — native Anthropic Messages API, with streaming. This is what Claude Code uses.
- **`POST /v1/chat/completions`** — OpenAI-compatible, for Cursor, Continue.dev, Copilot Chat, and any OpenAI SDK.

---

## Why

Claude Code (and most LLM coding tools) burn tokens faster than expected because:

1. **Context compounds** — every message re-sends the entire conversation history
2. **Tool outputs are verbose** — stack traces, grep results, and build logs dump thousands of tokens per call
3. **Files are re-injected in full** — even when only one line changed
4. **Tasks bleed into each other** — no automatic `/clear` between unrelated tasks
5. **Model overkill** — Opus-level model used for trivial rename tasks

ctx-gate addresses all five at the proxy layer, transparently.

---

## Architecture

```
Your IDE / Claude Code / CLI
        │
        ▼
┌─────────────────────────┐
│       ctx-gate          │  ← localhost:8080  (/v1/messages + /v1/chat/completions)
│                         │
│  ① Task Shift Detector  │  auto-clears context on new tasks
│  ② Context Compressor   │  summarizes old turns, keeps prompt-relevant ones,
│                         │  diffs files, strips noise, honors a token budget
│  ③ Model Router         │  fast/standard/advanced based on prompt complexity
│  ④ Checkpoint Writer    │  saves session state for restart recovery
└─────────────────────────┘
        │
        ▼
  Real LLM (Claude / OpenAI / Gemini / Ollama)
```

---

## Quick Start

### Install

```bash
git clone https://github.com/your-org/ctx-gate
cd ctx-gate
pip install fastapi uvicorn httpx

# Optional extras:
pip install "tiktoken"                  # exact token counts (else char/4 estimate)
pip install "lancedb sentence-transformers"   # RAG retrieval (else TF-IDF fallback)
```

### Start the proxy

```bash
# Claude (default)
ANTHROPIC_API_KEY=sk-ant-... python ctx_gate.py serve --verbose

# OpenAI
OPENAI_API_KEY=sk-... python ctx_gate.py serve --provider=openai

# Local Ollama (no key needed)
python ctx_gate.py serve --provider=ollama

# Custom port
python ctx_gate.py serve --port=9000
```

### Point your tool at ctx-gate

**Claude Code** (`~/.claude/settings.json`) — routed through the native `/v1/messages` endpoint, streaming included:
```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080"
  }
}
```
Claude Code sends its own API key/token through; ctx-gate forwards it upstream and falls back to the server's `ANTHROPIC_API_KEY` if none is present.

**Cursor / Continue.dev / VS Code**: Change the API base URL to `http://127.0.0.1:8080/v1`

**Any OpenAI SDK**:
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="any")
```

---

## MCP Integration (Claude Code native)

For tighter integration, register ctx-gate as a Claude Code MCP server:

```json
// ~/.claude/claude_desktop_config.json
{
  "mcpServers": {
    "ctx-gate": {
      "command": "python",
      "args": ["/path/to/ctx-gate/mcp_server.py"]
    }
  }
}
```

This exposes ctx-gate's tools directly to Claude:
- `compress_context` — compress a message list before sending
- `detect_task_shift` — check if a new prompt is a new task
- `write_checkpoint` — save session state
- `load_checkpoint` — restore from last checkpoint
- `route_model` — get recommended model tier for a prompt
- `get_stats` — token savings for current session

---

## What Each Module Does

### ① Task Shift Detector
Detects when you've moved to a new task using signal scoring:
- Explicit language: "now let's work on X", "switch to", "start fresh"
- File domain change (auth.ts → payments.go)
- Topic keyword cluster divergence

On shift: clears conversation history, extracts key facts (language, frameworks, files) to carry forward into the new session's system prompt.

### ② Context Compressor
Applied on every request. Savings are **per-strategy ceilings on the content they apply to**, not whole-session numbers — actual savings depend on how much of your session is old turns / verbose output (see [Faithfulness & Evaluation](#faithfulness--evaluation) for measured results):

| Strategy | Savings (on applicable content) |
|----------|---------|
| Rolling summary of turns older than `recency_window` | 40–70% |
| Relevance-scored retention (keep prompt-relevant old turns verbatim) | preserves facts the summary would drop |
| File diff injection (only send changed lines) | 80–95% on repeated file loads |
| Tool output truncation (keep first+last N lines) | 60–90% on verbose outputs |
| Large code block compression | 50–80% on pasted files |
| Adaptive token budget (`--token-budget`) | drops least-relevant turns to fit a hard cap |

**Relevance-scored retention** is what keeps compression honest: before summarizing old turns, the compressor keeps the few most relevant to your *current* prompt verbatim (bag-of-words overlap), so a fact you're now asking about isn't summarized away. This is why the bundled eval reports 100% fact-retention.

### ③ Model Router
Auto-selects model tier based on prompt complexity:

| Signal | Tier | Example |
|--------|------|---------|
| "typo", "rename", "comment", "what is" | `fast` (Haiku 4.5 / GPT-4o-mini) | Fix the spelling in line 42 |
| Default | `standard` (Sonnet 4.6 / GPT-4o) | Add a new API endpoint |
| "architecture", "cross-cutting", "refactor entire", "root cause" | `advanced` (Opus 4.8 / o1) | Redesign the auth system |
| Long context (>60k tokens) | at least `standard` | Any long session |

The routed tier is authoritative — the proxy rewrites the upstream model accordingly. Per-tier model IDs are overridable in `ModelRouter`. Force one tier for everything with `--model=advanced`.

### ④ Checkpoint Writer
A proxy can't observe individual tool-call events, so checkpoints are derived from the conversation snapshot on each request and written every N requests (configurable) to `.ctx-gate/`:
```json
{
  "task_description": "Build REST API",
  "decisions": ["Used FastAPI over Flask for async support"],
  "files_touched": ["src/main.py", "src/auth.py"],
  "next_steps": ["Add authentication middleware"],
  "turn_count": 23,
  "tool_call_count": 45
}
```
On session restart, the checkpoint is injected into the new system prompt automatically.

---

## CLI Reference

```bash
ctx-gate serve                          # proxy on :8080, Claude provider
ctx-gate serve --provider=openai        # use OpenAI (also: gemini, ollama)
ctx-gate serve --model=advanced         # force Opus/o1 for everything
ctx-gate serve --recency-window=10      # keep 10 recent turns verbatim
ctx-gate serve --token-budget=20000     # hard cap; drop least-relevant turns to fit
ctx-gate serve --llm-summary            # summarize old turns with the fast tier
ctx-gate serve --rag --project-root=.   # inject only semantically-relevant code chunks
ctx-gate serve --verbose                # log compression stats per request

ctx-gate status                         # show stats (proxy must be running)
ctx-gate compress "some long text"      # test compressor
ctx-gate detect-shift "new prompt"      # test task shift detection

ctx-gate eval                           # run the faithfulness harness (savings + retention)
ctx-gate eval --json                    # machine-readable report
ctx-gate eval --min-retention 1.0       # exit nonzero if any fact is dropped (CI gate)
ctx-gate eval --llm                     # also score answer accuracy (needs ANTHROPIC_API_KEY)
```

---

## Tuning

**`--recency-window`** (default: 6): Number of recent turns kept verbatim. Increase if you find Claude losing recent context; decrease for more aggressive compression.

**`--model`**: Force a tier. Use `advanced` for architecture sessions, `fast` for bulk scripting.

**`--token-budget`** (default: off): Hard cap on tokens per request. When set, ctx-gate drops the least prompt-relevant turns (never a system message or your current prompt) until the request fits.

**`--llm-summary`** (default: off): Summarize old turns with the fast-tier model instead of the local extractive summarizer. Higher-quality summaries at the cost of one extra fast-model call per request; falls back to extractive automatically if that call fails.

**`.claudeignore`**: Add to your project to prevent large directories from being indexed:
```
node_modules/
dist/
.git/
*.lock
__pycache__/
```

**CLAUDE.md size**: Every line in CLAUDE.md is prepended to every turn. Keep it under 2KB. ctx-gate will warn if it's larger.

---

## Optional: Accurate Token Counting

Token counts (and therefore reported savings) use `tiktoken` when it's installed, and fall back to a char/4 estimate otherwise. The `ctx-gate eval` report tells you which backed a given run (`token counts: accurate (tiktoken)` vs `estimated (char/4)`).

```bash
pip install tiktoken                    # auto-detected and used when present
```

## Optional: RAG-based Retrieval

For very large codebases, install the RAG extras to store file chunks in a vector DB:

```bash
pip install "ctx-gate[rag]"            # adds lancedb + sentence-transformers
```

Then pass `--rag --project-root=<dir>` to `ctx-gate serve`. The model receives only the chunks semantically relevant to each prompt, instead of full files. Without the extras, RAG still runs on a built-in TF-IDF + in-memory fallback (lower quality, zero extra dependencies).

---

## Faithfulness & Evaluation

"Reduce tokens without losing accuracy" is only credible if it's measured, so ctx-gate ships a harness that does exactly that. Run it any time:

```bash
$ ctx-gate eval
ctx-gate faithfulness report
============================================================
scenario                      savings  retention
------------------------------------------------------------
database-choice                  0.7%      100%
auth-mechanism                   0.4%      100%
rate-limit-gap                   0.4%      100%
long-session-logs               90.9%      100%
recent-constraint                0.7%      100%
------------------------------------------------------------
MEAN                            18.6%      100%
token counts: accurate (tiktoken)
```

Each scenario buries a fact in an early turn (the kind compression summarizes), pads the history, then probes for that fact. The harness measures two things:

- **Layer A — fact retention** (deterministic, no API key, CI-safe): after compression, do the facts the answer depends on still appear in the context the model would receive? This directly tests the compressor and is fully reproducible.
- **Layer B — answer accuracy** (`--llm`, opt-in): ask the model the probe with full vs. compressed context and score each answer. The *delta* is the real signal — ~0 means compression didn't change the answer.

The numbers tell an honest story rather than a marketing one: short Q&A barely compresses (0.4–0.7%), while a long session full of verbose logs compresses **90.9%** — and retention stays at 100% across the board because relevance-scored retention keeps the probed fact verbatim. Wire it into CI as a regression gate:

```bash
ctx-gate eval --min-retention 1.0   # exit nonzero if compression drops any required fact
```

The harness is designed to be able to *fail* the product's own claim — the test suite includes a case (relevance disabled) where a fact is dropped and the report flags it, so a passing report means something.

---

## Stats

```bash
$ ctx-gate status
{
  "session_id": "a3f8c1d2",
  "requests_proxied": 47,
  "tokens_saved_estimate": 83400,
  "task_shifts_detected": 3
}
```

---

## Status & Known Limitations

ctx-gate is **early but real** — every feature documented above is wired into the request path and covered by tests (`pytest -q` → 69 passing). What that does and doesn't mean:

**Working today**
- Native Anthropic `/v1/messages` (with streaming) and OpenAI `/v1/chat/completions`.
- **Streaming in both directions**, including OpenAI-format clients streaming against a Claude backend — the Anthropic SSE is translated to OpenAI `chat.completion.chunk` SSE on the fly (text and tool-call deltas).
- Task-shift clearing, context compression, relevance-scored retention, file-diff injection, tool-output truncation, model routing (routed model is applied upstream), token budgeting, snapshot checkpoints, optional RAG, optional LLM summary.
- A faithfulness harness with a CI gate.

**Known limitations / rough edges**
- **Checkpoints are snapshot-based, not event-based** — a proxy can't see individual tool calls, so checkpoint counts are derived from conversation history, not a live tool-call stream.
- **Relevance scoring is lexical** (bag-of-words overlap), not embedding-based. It reliably catches keyword-overlapping facts; paraphrased probes may need the RAG path.
- **`--llm-summary` adds latency** (one fast-model call per compressed request) and is off by default.
- **Stats are in-memory** and reset when the proxy restarts.
- Model IDs in `ModelRouter` are sensible defaults, not guaranteed current for every provider — override per tier as needed.

**Roadmap**
- Persistent stats + per-session isolation, upstream retry/resilience.
- Embedding-based relevance scoring shared with the RAG store.

---

## Development

```bash
pip install -e ".[dev,tokenizer]"      # editable install with test + tokenizer deps
pytest -q                              # run the suite (69 tests)
ctx-gate eval --min-retention 1.0      # run the faithfulness gate locally
```

CI (GitHub Actions, [`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the test suite **and** the faithfulness gate on every push and PR across Python 3.11–3.13, and uploads the faithfulness report as a build artifact. A change that drops a required fact fails CI — savings can't silently regress accuracy.

---

## License

MIT
