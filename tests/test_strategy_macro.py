"""Tests for the mid-term macro factor sleeve."""
import numpy as np
import pandas as pd

from strategy_macro import (
    below_ma_signal,
    build_factor_signals,
    falling_signal,
    flow_confirmation_signal,
    macro_composite_position,
    rising_signal,
)


def _frame(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    gold = pd.Series(2000 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))), index=idx)
    return pd.DataFrame(
        {
            "Gold": gold,
            "USD_Index": 100 + np.cumsum(rng.normal(0, 0.2, n)),
            "Real_10Y": 1.5 + np.cumsum(rng.normal(0, 0.02, n)),
            "Breakeven_10Y": 2.2 + np.cumsum(rng.normal(0, 0.01, n)),
            "GLD_DollarVolume": np.abs(rng.normal(2e9, 4e8, n)),
        },
        index=idx,
    )


def test_falling_and_rising_signals_are_causal_and_binary():
    series = pd.Series(np.linspace(10, 5, 200), index=pd.date_range("2023-01-01", periods=200))
    fall = falling_signal(series, 63)
    rise = rising_signal(series, 63)
    assert set(fall.unique()) <= {0.0, 1.0}
    # Strictly falling series: after warm-up, falling=1, rising=0.
    assert fall.iloc[100:].eq(1.0).all()
    assert rise.iloc[100:].eq(0.0).all()
    # Warm-up region must be flat 0, never NaN.
    assert fall.iloc[:63].eq(0.0).all()


def test_below_ma_signal_matches_definition():
    series = pd.Series([10.0] * 60 + [5.0] * 60, index=pd.date_range("2023-01-01", periods=120))
    sig = below_ma_signal(series, 50)
    assert sig.iloc[55] == 0.0  # flat at level -> not strictly below MA
    assert sig.iloc[65] == 1.0  # right after the drop, MA still carries the 10s
    assert sig.iloc[-1] == 0.0  # once the window is all 5s, price == MA again


def test_flow_confirmation_requires_both_conditions():
    idx = pd.date_range("2023-01-01", periods=200, freq="B")
    up_prices = pd.Series(np.linspace(1800, 2200, 200), index=idx)
    flat_volume = pd.Series(1e9, index=idx)
    sig = flow_confirmation_signal(up_prices, flat_volume)
    # Volume never expands (fast == slow) -> signal stays 0 despite uptrend.
    assert sig.eq(0.0).all()

    expanding_volume = pd.Series(np.linspace(1e9, 3e9, 200), index=idx)
    sig2 = flow_confirmation_signal(up_prices, expanding_volume)
    assert sig2.iloc[-20:].eq(1.0).all()


def test_build_factor_signals_uses_available_columns():
    frame, used = build_factor_signals(_frame())
    assert set(used) == {
        "real_rate_momentum",
        "usd_downtrend",
        "inflation_expectation",
        "flow_confirmation",
    }
    composite, used2 = macro_composite_position(_frame())
    assert used2 == used
    assert composite.dropna().between(0.0, 1.0).all()


def test_build_factor_signals_degrades_to_trend_fallback():
    df = _frame()[["Gold"]]
    frame, used = build_factor_signals(df)
    assert used == ["price_trend_fallback"]
    assert frame["price_trend_fallback"].isin([0.0, 1.0]).all()
