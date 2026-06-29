"""Tests for the faithfulness harness (Phase 1)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval import FaithfulnessHarness, FaithfulnessReport, Scenario, SCENARIOS
from src.compressor.compressor import ContextCompressor


def _by_name(results, name):
    return next(r for r in results if r.name == name)


# A deterministic "perfect reader": its answer is just the context it received,
# so its answer-accuracy exactly tracks what survived compression. This lets us
# exercise Layer B end-to-end with no API key.
def perfect_reader(messages):
    parts = []
    for m in messages:
        c = m.get("content", "")
        parts.append(c if isinstance(c, str) else "")
    return "\n".join(parts)


class TestLayerARetention:

    def test_retained_fact_scenario(self):
        h = FaithfulnessHarness(recency_window=6)
        r = h.run(_scenario("database-choice"))
        assert r.fact_retention == 1.0
        assert r.dropped_facts == []

    def test_harness_can_detect_a_gap(self):
        # The harness must be able to FAIL the product's own claim, not rubber-stamp.
        # With relevance retention disabled, the rate-limit fact is summarized away.
        comp = ContextCompressor(recency_window=6, relevance_keep=0)
        r = FaithfulnessHarness(compressor=comp).run(_scenario("rate-limit-gap"))
        assert r.fact_retention < 1.0
        assert r.dropped_facts

    def test_relevance_closes_the_gap(self):
        # Phase 2: relevance-scored retention (default on) preserves the fact the
        # probe asks about, so the previously-known gap is closed.
        r = FaithfulnessHarness(recency_window=6).run(_scenario("rate-limit-gap"))
        assert r.fact_retention == 1.0
        assert r.dropped_facts == []

    def test_long_session_high_savings_full_retention(self):
        h = FaithfulnessHarness(recency_window=6)
        r = h.run(_scenario("long-session-logs"))
        assert r.savings_pct > 50           # verbose logs compress hard
        assert r.fact_retention == 1.0      # but the key fact survives

    def test_recent_fact_always_retained(self):
        h = FaithfulnessHarness(recency_window=6)
        r = h.run(_scenario("recent-constraint"))
        assert r.fact_retention == 1.0


class TestLayerBAccuracy:

    def test_no_loss_when_facts_retained(self):
        h = FaithfulnessHarness(recency_window=6)
        r = h.run(_scenario("database-choice"), model_fn=perfect_reader)
        assert r.accuracy_delta == 0.0      # compression didn't change the answer

    def test_accuracy_drops_when_facts_dropped(self):
        # With relevance disabled the fact is dropped, so even a perfect reader
        # loses accuracy on the compressed context — proving Layer B is sensitive.
        comp = ContextCompressor(recency_window=6, relevance_keep=0)
        r = FaithfulnessHarness(compressor=comp).run(
            _scenario("rate-limit-gap"), model_fn=perfect_reader)
        assert r.accuracy_delta is not None and r.accuracy_delta < 0

    def test_no_accuracy_loss_with_relevance(self):
        # Default (relevance on): compressed answer matches full-context answer.
        r = FaithfulnessHarness(recency_window=6).run(
            _scenario("rate-limit-gap"), model_fn=perfect_reader)
        assert r.accuracy_delta == 0.0


class TestReport:

    def test_run_all_aggregates(self):
        h = FaithfulnessHarness(recency_window=6)
        report = h.run_all(SCENARIOS)
        assert isinstance(report, FaithfulnessReport)
        assert 0.0 <= report.mean_retention <= 1.0
        assert report.mean_savings >= 0.0
        assert report.to_dict()["scenarios"] == len(SCENARIOS)

    def test_summary_is_ascii_safe(self):
        # Regression: summary must render on Windows cp1252 consoles.
        h = FaithfulnessHarness(recency_window=6)
        text = h.run_all(SCENARIOS, model_fn=perfect_reader).summary()
        text.encode("cp1252")  # must not raise

    def test_savings_field_never_negative(self):
        h = FaithfulnessHarness(recency_window=6)
        for r in h.run_all(SCENARIOS).results:
            assert r.savings_pct >= 0.0


def _scenario(name: str) -> Scenario:
    return _by_name(SCENARIOS, name)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
