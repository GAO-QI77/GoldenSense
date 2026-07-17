"""Tests for the cross-asset context block."""
import numpy as np
import pandas as pd
import pytest

from cross_asset import build_cross_asset_context


def _frame(days: int = 400) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=days)
    rng = np.random.default_rng(3)
    gold_ret = rng.normal(0.0004, 0.01, days)
    return pd.DataFrame({
        "Gold": 2000 * np.cumprod(1 + gold_ret),
        # Silver = gold move + small noise -> strongly positive correlation.
        "Silver": 25 * np.cumprod(1 + gold_ret + rng.normal(0, 0.002, days)),
        # USD = inverse of gold move -> strongly negative correlation.
        "USD_Index": 100 * np.cumprod(1 - gold_ret + rng.normal(0, 0.002, days)),
        "S&P500": 5000 * np.cumprod(1 + rng.normal(0.0004, 0.011, days)),
        "Crude_Oil": 75 * np.cumprod(1 + rng.normal(0.0, 0.02, days)),
        "10Y_Bond": 100 * np.cumprod(1 + rng.normal(0.0, 0.004, days)),
    }, index=idx)


def test_correlation_signs_and_performance():
    ctx = build_cross_asset_context(_frame())
    assert ctx is not None
    assert ctx["peers"]["Silver"]["corr_63d"] > 0.8
    assert ctx["peers"]["USD_Index"]["corr_63d"] < -0.8
    # 1y performance matches the price ratio to display rounding.
    frame = _frame()
    expected = frame["Gold"].iloc[-1] / frame["Gold"].iloc[-252] - 1.0
    assert ctx["gold_perf_1y"] == pytest.approx(expected, abs=5e-4)
    assert ctx["note"]


def test_missing_peer_degrades_per_asset():
    frame = _frame().drop(columns=["Crude_Oil"])
    ctx = build_cross_asset_context(frame)
    assert "Crude_Oil" in ctx["missing"]
    assert "Silver" in ctx["peers"]


def test_missing_gold_returns_none():
    frame = _frame().drop(columns=["Gold"])
    assert build_cross_asset_context(frame) is None
