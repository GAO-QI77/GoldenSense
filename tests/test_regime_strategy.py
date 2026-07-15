"""Tests for the regime decision brain.

This module turns the BACKTESTED signals (multi-timescale trend + vol management)
into a stance/action/exposure per risk profile. These tests pin the mapping that
the agent gateway will rely on, so the product's recommendations stay tied to
something reproducible rather than a fake probability.
"""
import numpy as np
import pandas as pd

from regime_strategy import evaluate_regime


def _rising(n=300, start=1500.0, step=2.0):
    return pd.Series(start + step * np.arange(n), index=pd.date_range("2021-01-01", periods=n))


def _falling(n=300, start=2500.0, step=2.0):
    return pd.Series(start - step * np.arange(n), index=pd.date_range("2021-01-01", periods=n))


def test_strong_uptrend_is_bullish_with_full_trend_score():
    d = evaluate_regime(_rising(), risk_profile="balanced", vol_state="calm")
    assert d.regime == "uptrend"
    assert d.stance == "偏多"
    assert d.trend_score == 1.0
    assert d.confidence_band == "高"


def test_strong_downtrend_is_bearish_and_cuts_exposure():
    d = evaluate_regime(_falling(), risk_profile="balanced", vol_state="calm")
    assert d.regime == "downtrend"
    assert d.stance == "偏空"
    assert d.action == "降低暴露"
    assert d.trend_score == 0.0
    assert d.target_exposure_pct == 0.0


def test_conservative_holds_less_than_aggressive_in_same_uptrend():
    cons = evaluate_regime(_rising(), risk_profile="conservative", vol_state="calm")
    aggr = evaluate_regime(_rising(), risk_profile="aggressive", vol_state="calm")
    assert cons.target_exposure_pct < aggr.target_exposure_pct


def test_stress_volatility_reduces_exposure():
    calm = evaluate_regime(_rising(), risk_profile="aggressive", vol_state="calm")
    stress = evaluate_regime(_rising(), risk_profile="aggressive", vol_state="stress")
    assert stress.target_exposure_pct < calm.target_exposure_pct
    assert stress.confidence_band == "低"


def test_insufficient_history_degrades_to_low_confidence_without_crashing():
    short = pd.Series([1900.0, 1910.0, 1905.0], index=pd.date_range("2024-01-01", periods=3))
    d = evaluate_regime(short, risk_profile="balanced", vol_state=None)
    assert d.confidence_band == "低"
    assert d.regime in {"uptrend", "downtrend", "mixed"}


def test_accepts_plain_sequence_not_just_series():
    prices = [1500.0 + 2.0 * i for i in range(300)]
    d = evaluate_regime(prices, risk_profile="balanced", vol_state="calm")
    assert d.regime == "uptrend"
    assert d.stance == "偏多"


def test_exposure_is_bounded_0_to_100():
    for profile in ("conservative", "balanced", "aggressive"):
        for vol in ("calm", "elevated", "stress"):
            d = evaluate_regime(_rising(), risk_profile=profile, vol_state=vol)
            assert 0.0 <= d.target_exposure_pct <= 100.0
