"""
ModelRouter: Automatically selects the right model tier for each task.

Simple → Haiku (cheapest)
Standard coding → Sonnet (default)
Complex architecture / multi-file refactor → Opus

This alone can cut token *cost* dramatically without reducing output quality,
since you're not burning Opus-level quota on routine edits.
"""

import re
from dataclasses import dataclass


# Claude model tiers (also works with OpenAI/Gemini equivalents).
# These are sensible defaults; override per tier via ModelRouter(tier_overrides=...).
CLAUDE_TIERS = {
    "fast":     "claude-haiku-4-5-20251001",
    "standard": "claude-sonnet-4-6",
    "advanced": "claude-opus-4-8",
}

OPENAI_TIERS = {
    "fast":     "gpt-4o-mini",
    "standard": "gpt-4o",
    "advanced": "o1",
}

GEMINI_TIERS = {
    "fast":     "gemini-1.5-flash",
    "standard": "gemini-1.5-pro",
    "advanced": "gemini-1.5-pro",
}

PROVIDER_TIERS = {
    "claude":  CLAUDE_TIERS,
    "openai":  OPENAI_TIERS,
    "gemini":  GEMINI_TIERS,
}

# Task complexity classifiers
ADVANCED_SIGNALS = [
    r"\b(architect|architecture|design (system|pattern)|redesign)\b",
    r"\b(refactor (entire|whole|all|the entire))\b",
    r"\b(cross.cutting|multi.?file|across (the )?codebase)\b",
    r"\b(performance (bottleneck|profil|optim))\b",
    r"\b(security (audit|review|vulnerabilit))\b",
    r"\b(why (is|does|isn'?t|doesn'?t).*broken|root cause)\b",
    r"\b(tradeoff|trade.off|pros and cons|compare approaches)\b",
]

FAST_SIGNALS = [
    r"\b(rename|typo|spelling|format|lint|whitespace)\b",
    r"\b(add (a )?(comment|docstring|type hint|import))\b",
    r"\b(what (is|does|are)|explain|define|translate)\b",
    r"\b(simple (fix|change|edit|update))\b",
    r"\b(quick (check|look|question))\b",
    r"\b(generate (a )?(boilerplate|scaffold|template|stub))\b",
]

_advanced_re = re.compile("|".join(ADVANCED_SIGNALS), re.IGNORECASE)
_fast_re = re.compile("|".join(FAST_SIGNALS), re.IGNORECASE)


@dataclass
class RoutingDecision:
    tier: str           # "fast" | "standard" | "advanced"
    model: str          # actual model string
    reason: str
    confidence: float


class ModelRouter:
    """
    Routes each prompt to an appropriate model tier.

    Can be overridden by the user with explicit flags:
      ctx-gate --model=advanced "redesign the auth system"
    """

    def __init__(
        self,
        provider: str = "claude",
        default_tier: str = "standard",
        force_tier: str | None = None,
        tier_overrides: dict[str, str] | None = None,
    ):
        self.provider = provider
        self.default_tier = default_tier
        self.force_tier = force_tier
        # Whether this provider has a known tier->model map. Providers that don't
        # (e.g. Ollama and other local/custom backends) keep the client's chosen
        # model — see `routes_models`. Tier classification still runs for logging.
        self.routes_models = provider in PROVIDER_TIERS or bool(tier_overrides)
        self.tiers = dict(PROVIDER_TIERS.get(provider, CLAUDE_TIERS))
        if tier_overrides:
            self.tiers.update(tier_overrides)

    def route(self, prompt: str, context_length_tokens: int = 0) -> RoutingDecision:
        """Decide which model tier to use for this prompt."""
        if self.force_tier:
            return RoutingDecision(
                tier=self.force_tier,
                model=self.tiers[self.force_tier],
                reason="forced by user flag",
                confidence=1.0,
            )

        tier, reason, confidence = self._classify(prompt, context_length_tokens)
        return RoutingDecision(
            tier=tier,
            model=self.tiers[tier],
            reason=reason,
            confidence=confidence,
        )

    def get_model(self, prompt: str, context_length_tokens: int = 0) -> str:
        return self.route(prompt, context_length_tokens).model

    # ------------------------------------------------------------------

    def _classify(self, prompt: str, ctx_tokens: int) -> tuple[str, str, float]:
        adv_match = _advanced_re.search(prompt)
        fast_match = _fast_re.search(prompt)

        # Long context → bump up to standard at minimum (Haiku degrades on long ctx)
        if ctx_tokens > 60_000 and not adv_match:
            return "standard", f"long context ({ctx_tokens} tokens)", 0.7

        if adv_match and not fast_match:
            return "advanced", f"advanced signal: '{adv_match.group()}'", 0.8

        if fast_match and not adv_match:
            return "fast", f"simple task signal: '{fast_match.group()}'", 0.75

        return self.default_tier, "no strong signal, using default", 0.5
