"""
Faithfulness harness for ctx-gate.

The product claim is "reduce tokens *without losing accuracy*." That claim is
only credible if it's measured. This harness measures it two ways:

  Layer A — Fact retention (deterministic, no API key, CI-safe):
      After compression, do the facts a correct answer depends on still appear
      anywhere in the context the model would receive? This directly tests the
      compressor and is fully reproducible.

  Layer B — Answer accuracy (LLM-in-the-loop, opt-in):
      Ask the model the probe question with full context vs. compressed context.
      Score each answer against the required facts. The *delta* (compressed minus
      full) is the real faithfulness signal: ~0 means compression didn't hurt.

A scenario buries facts in early turns (the ones compression summarizes), then
probes for them — so any information loss shows up as a measurable drop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.compressor.compressor import ContextCompressor
from src.compressor.tokens import TokenCounter


# A model function takes an OpenAI-format message list and returns the answer text.
ModelFn = Callable[[list[dict]], str]


@dataclass
class Scenario:
    """One faithfulness test case."""
    name: str
    messages: list[dict]            # the conversation history (facts buried here)
    probe: str                      # question whose answer depends on buried facts
    required_facts: list[str]       # regex (or literal) each correct answer needs
    description: str = ""


@dataclass
class ScenarioResult:
    name: str
    full_tokens: int
    compressed_tokens: int
    savings_pct: float
    fact_retention: float                 # Layer A: 0.0–1.0
    retained_facts: list[str]
    dropped_facts: list[str]
    accurate_tokens: bool                 # were token counts tiktoken-backed?
    # Layer B (populated only when a model_fn is supplied)
    full_answer_score: Optional[float] = None
    compressed_answer_score: Optional[float] = None
    accuracy_delta: Optional[float] = None
    full_answer: Optional[str] = None
    compressed_answer: Optional[str] = None


@dataclass
class FaithfulnessReport:
    results: list[ScenarioResult] = field(default_factory=list)

    @property
    def mean_savings(self) -> float:
        return _mean([r.savings_pct for r in self.results])

    @property
    def mean_retention(self) -> float:
        return _mean([r.fact_retention for r in self.results])

    @property
    def mean_accuracy_delta(self) -> Optional[float]:
        deltas = [r.accuracy_delta for r in self.results if r.accuracy_delta is not None]
        return _mean(deltas) if deltas else None

    @property
    def has_llm_scores(self) -> bool:
        return any(r.accuracy_delta is not None for r in self.results)

    def worst_retention(self) -> list[ScenarioResult]:
        return sorted(self.results, key=lambda r: r.fact_retention)

    def to_dict(self) -> dict:
        return {
            "scenarios": len(self.results),
            "mean_savings_pct": round(self.mean_savings, 1),
            "mean_fact_retention": round(self.mean_retention, 3),
            "mean_accuracy_delta": (
                round(self.mean_accuracy_delta, 3)
                if self.mean_accuracy_delta is not None else None
            ),
            "results": [r.__dict__ for r in self.results],
        }

    def summary(self) -> str:
        lines = []
        lines.append("ctx-gate faithfulness report")
        lines.append("=" * 60)
        header = f"{'scenario':<28}{'savings':>9}{'retention':>11}"
        if self.has_llm_scores:
            header += f"{'acc-delta':>10}"
        lines.append(header)
        lines.append("-" * 60)
        for r in self.results:
            row = f"{r.name[:27]:<28}{r.savings_pct:>8.1f}%{r.fact_retention:>10.0%} "
            if r.accuracy_delta is not None:
                row += f"{r.accuracy_delta:>+9.0%}"
            lines.append(row)
            if r.dropped_facts:
                lines.append(f"    [!] dropped: {', '.join(r.dropped_facts)}")
        lines.append("-" * 60)
        tail = f"{'MEAN':<28}{self.mean_savings:>8.1f}%{self.mean_retention:>10.0%}"
        if self.mean_accuracy_delta is not None:
            tail += f"{self.mean_accuracy_delta:>+9.0%}"
        lines.append(tail)
        if not self.has_llm_scores:
            lines.append("(Layer A only -- run with a model_fn / --llm for answer-accuracy.)")
        acc = "accurate (tiktoken)" if (self.results and self.results[0].accurate_tokens) else "estimated (char/4)"
        lines.append(f"token counts: {acc}")
        return "\n".join(lines)


class FaithfulnessHarness:
    def __init__(self, compressor: ContextCompressor | None = None, recency_window: int = 6):
        self.compressor = compressor or ContextCompressor(recency_window=recency_window)
        self.counter = TokenCounter()

    def run(self, scenario: Scenario, model_fn: ModelFn | None = None) -> ScenarioResult:
        probe_msg = {"role": "user", "content": scenario.probe}
        full_messages = scenario.messages + [probe_msg]

        # Compress the full conversation (the probe is recent, so it survives).
        comp = self.compressor.compress(full_messages, scenario.probe)
        compressed_messages = comp.messages

        full_tokens = self.counter.count_messages(full_messages)
        compressed_tokens = self.counter.count_messages(compressed_messages)
        savings = _pct(full_tokens, compressed_tokens)

        # Layer A — fact retention in the compressed context.
        ctx_text = _join_text(compressed_messages)
        retained = [f for f in scenario.required_facts if _fact_present(f, ctx_text)]
        dropped = [f for f in scenario.required_facts if f not in retained]
        retention = len(retained) / max(1, len(scenario.required_facts))

        result = ScenarioResult(
            name=scenario.name,
            full_tokens=full_tokens,
            compressed_tokens=compressed_tokens,
            savings_pct=round(savings, 1),
            fact_retention=round(retention, 3),
            retained_facts=retained,
            dropped_facts=dropped,
            accurate_tokens=self.counter.accurate,
        )

        # Layer B — answer accuracy with vs. without compression.
        if model_fn is not None:
            full_answer = model_fn(full_messages)
            comp_answer = model_fn(compressed_messages)
            full_score = _answer_score(full_answer, scenario.required_facts)
            comp_score = _answer_score(comp_answer, scenario.required_facts)
            result.full_answer = full_answer
            result.compressed_answer = comp_answer
            result.full_answer_score = round(full_score, 3)
            result.compressed_answer_score = round(comp_score, 3)
            result.accuracy_delta = round(comp_score - full_score, 3)

        return result

    def run_all(self, scenarios: list[Scenario], model_fn: ModelFn | None = None) -> FaithfulnessReport:
        return FaithfulnessReport(results=[self.run(s, model_fn) for s in scenarios])


# ── Matching / scoring helpers ────────────────────────────────────────────────

def _fact_present(fact: str, text: str) -> bool:
    """Treat each required fact as a regex; fall back to literal substring."""
    try:
        return re.search(fact, text, re.IGNORECASE) is not None
    except re.error:
        return fact.lower() in text.lower()


def _answer_score(answer: str, required_facts: list[str]) -> float:
    if not required_facts:
        return 1.0
    hits = sum(1 for f in required_facts if _fact_present(f, answer or ""))
    return hits / len(required_facts)


def _join_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(b.get("text", "") for b in c if isinstance(b, dict))
    return "\n".join(parts)


def _pct(full: int, compressed: int) -> float:
    if full <= 0:
        return 0.0
    return max(0.0, (full - compressed) / full) * 100


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ── Optional Anthropic-backed model function ──────────────────────────────────

def make_anthropic_model_fn(model: str = "claude-sonnet-4-6", max_tokens: int = 512) -> ModelFn:
    """
    Build a model_fn that calls the Anthropic Messages API. Requires
    ANTHROPIC_API_KEY in the environment. Used for Layer B (`--llm`).
    """
    import os
    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — required for LLM faithfulness scoring.")

    def model_fn(messages: list[dict]) -> str:
        system = "\n\n".join(
            _join_text([m]) for m in messages if m.get("role") == "system"
        )
        conv = [
            {"role": m["role"], "content": _join_text([m])}
            for m in messages if m.get("role") in ("user", "assistant")
        ]
        body = {"model": model, "max_tokens": max_tokens, "messages": conv}
        if system:
            body["system"] = system
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            json=body,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    return model_fn
