"""
Built-in faithfulness scenarios.

Each scenario buries facts in early turns (which compression summarizes), pads
the middle, then probes for those facts at the end. Fact placement is chosen to
exercise different compressor behaviors:

  - facts in a user message's first sentence      → kept by extractive summary
  - facts in an assistant "signal" sentence        → kept (decision/file/verb match)
  - facts in a plain assistant sentence            → at risk of being dropped
  - facts in a recent turn                          → always kept (verbatim window)

The "rate-limit" scenario intentionally buries a fact where the current
extractive summarizer does NOT keep it, so the report surfaces a real retention
gap instead of rubber-stamping. That is a feature: the harness should be able to
fail the product's own claim.
"""

from __future__ import annotations

from .harness import Scenario


def _filler(n: int) -> list[dict]:
    """Generic back-and-forth turns to push early facts out of the recency window."""
    out = []
    topics = ["the build script", "the lint config", "the test runner",
              "the CI matrix", "the changelog", "the dev container",
              "the pre-commit hook", "the makefile target"]
    for i in range(n):
        t = topics[i % len(topics)]
        out.append({"role": "user", "content": f"Can you tweak {t}? Minor cleanup."})
        out.append({"role": "assistant", "content": f"Updated {t}; no functional change."})
    return out


SCENARIOS: list[Scenario] = [
    Scenario(
        name="database-choice",
        description="Fact in a user first-sentence — should be retained by summary.",
        messages=(
            [{"role": "user",
              "content": "We're standardizing on PostgreSQL as the primary datastore "
                         "because we need JSONB columns. Set up the connection pool."},
             {"role": "assistant",
              "content": "Done — I created db/pool.py with an async PostgreSQL pool."}]
            + _filler(8)
        ),
        probe="Remind me which database we settled on and why?",
        required_facts=[r"postgres", r"jsonb"],
    ),

    Scenario(
        name="auth-mechanism",
        description="Decision in an assistant 'signal' sentence (refactored + file).",
        messages=(
            [{"role": "user", "content": "Make the login flow stateless."},
             {"role": "assistant",
              "content": "I refactored auth.py to issue JWT access tokens signed with RS256."}]
            + _filler(8)
        ),
        probe="What auth mechanism and signing algorithm did we go with?",
        required_facts=[r"JWT", r"RS256"],
    ),

    Scenario(
        name="rate-limit-gap",
        description="Fact in a plain assistant sentence — KNOWN retention gap.",
        messages=(
            [{"role": "user", "content": "Add basic request throttling to the API."},
             {"role": "assistant",
              "content": "Added throttling middleware. The rate limit is 100 requests "
                         "per minute per API key. It returns HTTP 429 when exceeded."}]
            + _filler(8)
        ),
        probe="What's the configured rate limit?",
        required_facts=[r"100 requests", r"429"],
    ),

    Scenario(
        name="long-session-logs",
        description="Verbose tool output + long history: high savings, fact retained.",
        messages=(
            [{"role": "user",
              "content": "We deploy to the Kubernetes namespace 'payments-prod'. "
                         "A pod is crash-looping — investigate."},
             {"role": "assistant",
              "content": "I inspected the deployment and fixed the readiness probe in deploy.yaml."},
             {"role": "tool", "name": "kubectl",
              "content": "\n".join(f"[pod payments-api] log line {i}: handling request id={i}"
                                    for i in range(300))}]
            + _filler(6)
            + [{"role": "tool", "name": "pytest",
                "content": "\n".join(f"tests/test_pay_{i}.py ... PASSED" for i in range(200))}]
        ),
        probe="Which Kubernetes namespace do we deploy to?",
        required_facts=[r"payments-prod"],
    ),

    Scenario(
        name="recent-constraint",
        description="Fact in a recent turn — must survive (verbatim window).",
        messages=(
            _filler(8)
            + [{"role": "user",
                "content": "Important: the deploy target is AWS Lambda with a 15-minute "
                           "timeout, so keep cold starts small."},
               {"role": "assistant",
                "content": "Understood — I'll avoid heavy imports at module load."}]
        ),
        probe="What's our deploy target and its timeout?",
        required_facts=[r"Lambda", r"15.minute"],
    ),
]
