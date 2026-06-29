#!/usr/bin/env python3
"""
ctx-gate CLI

Usage:
  ctx-gate serve                          # start proxy on port 8080
  ctx-gate serve --provider=openai       # use OpenAI as backend
  ctx-gate serve --port=9000 --verbose   # custom port with logs
  ctx-gate status                        # show proxy stats
  ctx-gate compress "path/to/file"       # test compressor on a file
  ctx-gate detect-shift "new prompt"     # test task shift detection
"""

import sys
import argparse
import json
from pathlib import Path

# Force UTF-8 console output so banners/logs don't crash on legacy code pages
# (e.g. Windows cp1252 can't encode characters like "->" arrows or bullets).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).parent))


def cmd_serve(args):
    from src.proxy.server import run_server
    run_server(
        provider=args.provider,
        port=args.port,
        host=args.host,
        force_tier=args.model,
        recency_window=args.recency_window,
        verbose=args.verbose,
        rag=args.rag,
        project_root=args.project_root,
        token_budget=args.token_budget,
        llm_summary=args.llm_summary,
        max_sessions=args.max_sessions,
        max_retries=args.retries,
    )


def cmd_status(args):
    try:
        import httpx
        r = httpx.get(f"http://{args.host}:{args.port}/stats")
        data = r.json()
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Could not reach proxy at {args.host}:{args.port} — {e}")
        print("Start it with: ctx-gate serve")


def cmd_compress(args):
    from src.compressor.compressor import ContextCompressor

    text = Path(args.input).read_text() if Path(args.input).exists() else args.input
    compressor = ContextCompressor(recency_window=args.recency_window)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Previous long context here.\n" + ("word " * 500)},
        {"role": "assistant", "content": "Previous response.\n" + ("response " * 300)},
        {"role": "user", "content": text},
    ]
    result = compressor.compress(messages, text)
    print(f"Original:   {result.original_tokens:,} tokens")
    print(f"Compressed: {result.compressed_tokens:,} tokens")
    print(f"Saved:      {result.savings_pct}%")
    print(f"Summary:    {'injected' if result.summary_injected else 'not needed'}")


def cmd_detect_shift(args):
    from src.compressor.task_shift import TaskShiftDetector
    detector = TaskShiftDetector()
    result = detector.detect(args.prompt, [])
    print(f"Task shift: {'YES' if result.is_shift else 'NO'}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reason:     {result.reason}")
    if result.suggested_carry_forward:
        print(f"Carry fwd:  {result.suggested_carry_forward}")


def cmd_eval(args):
    from src.eval import FaithfulnessHarness, SCENARIOS, make_anthropic_model_fn

    model_fn = None
    if args.llm:
        try:
            model_fn = make_anthropic_model_fn(model=args.eval_model)
        except Exception as e:
            print(f"Could not enable LLM scoring: {e}")
            return 1

    harness = FaithfulnessHarness(recency_window=args.recency_window)
    report = harness.run_all(SCENARIOS, model_fn=model_fn)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    # Optional gate: fail (nonzero exit) if compression drops too many facts.
    if args.min_retention is not None and report.mean_retention < args.min_retention:
        print(f"\nFAIL: mean retention {report.mean_retention:.0%} "
              f"< required {args.min_retention:.0%}")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="ctx-gate",
        description="LLM-agnostic context optimization proxy",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)

    sub = parser.add_subparsers(dest="command")

    # serve
    p_serve = sub.add_parser("serve", help="Start the proxy server")
    p_serve.add_argument("--provider", default="claude",
                         choices=["claude", "openai", "gemini", "ollama"],
                         help="LLM provider to route to")
    p_serve.add_argument("--model", default=None,
                         choices=["fast", "standard", "advanced"],
                         help="Force a model tier (overrides auto-routing)")
    p_serve.add_argument("--recency-window", type=int, default=6,
                         help="Number of recent turns to keep verbatim (default: 6)")
    p_serve.add_argument("--verbose", "-v", action="store_true",
                         help="Log compression stats per request")
    p_serve.add_argument("--rag", action="store_true",
                         help="Enable RAG retrieval: inject only relevant code chunks")
    p_serve.add_argument("--project-root", default=".",
                         help="Project directory to index for RAG (default: cwd)")
    p_serve.add_argument("--token-budget", type=int, default=None,
                         help="Hard cap on tokens per request; drops least-relevant turns to fit")
    p_serve.add_argument("--llm-summary", action="store_true",
                         help="Summarize old turns with the fast-tier model (vs extractive)")
    p_serve.add_argument("--max-sessions", type=int, default=128,
                         help="Max isolated client sessions to keep (LRU-evicted)")
    p_serve.add_argument("--retries", type=int, default=2,
                         help="Retry attempts for transient upstream failures (default: 2)")
    p_serve.set_defaults(func=cmd_serve)

    # status
    p_status = sub.add_parser("status", help="Show proxy stats")
    p_status.set_defaults(func=cmd_status)

    # compress (test mode)
    p_compress = sub.add_parser("compress", help="Test compressor on text or a file")
    p_compress.add_argument("input", help="Text or file path")
    p_compress.add_argument("--recency-window", type=int, default=6)
    p_compress.set_defaults(func=cmd_compress)

    # detect-shift (test mode)
    p_shift = sub.add_parser("detect-shift", help="Test task shift detection")
    p_shift.add_argument("prompt", help="Prompt to analyze")
    p_shift.set_defaults(func=cmd_detect_shift)

    # eval (faithfulness harness)
    p_eval = sub.add_parser("eval", help="Run the faithfulness harness")
    p_eval.add_argument("--recency-window", type=int, default=6)
    p_eval.add_argument("--llm", action="store_true",
                        help="Enable Layer B answer-accuracy scoring (needs ANTHROPIC_API_KEY)")
    p_eval.add_argument("--eval-model", default="claude-sonnet-4-6",
                        help="Model to use for --llm scoring")
    p_eval.add_argument("--min-retention", type=float, default=None,
                        help="Exit nonzero if mean fact-retention falls below this (0-1)")
    p_eval.add_argument("--json", action="store_true", help="Emit the report as JSON")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    rc = args.func(args)
    if isinstance(rc, int):
        sys.exit(rc)


if __name__ == "__main__":
    main()
