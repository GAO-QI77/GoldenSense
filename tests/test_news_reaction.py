"""Tests for the analog-prior news analyst and event-driven cache invalidation."""
import time

import pytest

from analyst_committee import build_committee, news_analyst
from research_context import LocalQuantContext

ANALOG_PRIOR = {
    "category": "monetary_policy",
    "n_30d": 19,
    "mean_30d": 0.036,
    "positive_share_30d": 0.74,
    "matched_n_30d": 8,
    "matched_mean_excess_30d": 0.013,
}


def test_news_analyst_reports_matched_analogs_without_shifting_stance():
    base = news_analyst(0.0, 15.0)
    with_prior = news_analyst(0.0, 15.0, analog_prior=ANALOG_PRIOR)
    assert with_prior.stance == base.stance
    assert with_prior.confidence == base.confidence
    assert "8 次事件" in " ".join(with_prior.evidence)
    assert "+1.3%" in " ".join(with_prior.evidence)


def test_news_analyst_does_not_tilt_stance_using_unmatched_returns():
    raw = dict(ANALOG_PRIOR, matched_n_30d=0, matched_mean_excess_30d=None)
    assert news_analyst(0.0, 15.0, analog_prior=raw).stance == news_analyst(0.0, 15.0).stance


def test_news_analyst_vix_gate_still_caps_after_prior():
    view = news_analyst(0.5, 35.0, analog_prior=ANALOG_PRIOR)
    assert view.stance <= 0.0  # circuit breaker outranks any prior


def test_build_committee_passes_analog_prior():
    regime = {"trend_score": 0.6, "regime": "uptrend", "vol_state": "calm",
              "target_exposure_pct": 50, "sufficient_history": True}
    without = build_committee(regime=regime, news_sentiment=0.0, vix_value=15.0)
    with_prior = build_committee(
        regime=regime, news_sentiment=0.0, vix_value=15.0,
        analog_prior=ANALOG_PRIOR,
    )
    news_a = next(v for v in with_prior["views"] if v["name"] == "news")
    news_b = next(v for v in without["views"] if v["name"] == "news")
    assert news_a["stance"] == news_b["stance"]
    assert any("+1.3%" in line for line in news_a["evidence"])


# --------------------------------------------------------------------------- #
def test_research_context_invalidate():
    ctx = LocalQuantContext(ttl_seconds=3600)
    ctx._cache = {"sentinel": True}
    ctx._computed_at = time.time()
    # Younger than min age -> refused (anti-thrash guard).
    assert ctx.invalidate(min_age_seconds=9999) is False
    assert ctx._cache is not None
    # Old enough -> invalidated.
    assert ctx.invalidate(min_age_seconds=0) is True
    assert ctx._cache is None
    # Nothing cached -> no-op.
    assert ctx.invalidate() is False
