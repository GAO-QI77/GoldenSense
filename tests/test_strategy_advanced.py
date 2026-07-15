"""Tests for the advanced (multi-timescale + cost-aware) strategy primitives.

Same non-negotiable property as before: no look-ahead. Plus the turnover-control
behaviour of the rebalancing band, which is what makes vol-management deployable.
"""
import numpy as np
import pandas as pd

from strategy_advanced import multi_trend_signal, rebalance_band


def _idx(n):
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_multi_trend_averages_subsignals():
    # At the last bar: above the fast MA (long) but below the slow MA (flat).
    prices = pd.Series([10.0, 20.0, 10.0, 10.0, 10.5], index=_idx(5))
    sig = multi_trend_signal(prices, lookbacks=(2, 4))
    # fast(2) -> 1, slow(4) -> 0  =>  average 0.5
    assert sig.iloc[-1] == 0.5


def test_multi_trend_full_long_when_all_subsignals_long():
    rising = pd.Series(np.arange(1, 21, dtype=float), index=_idx(20))
    sig = multi_trend_signal(rising, lookbacks=(2, 5, 10))
    assert sig.iloc[-1] == 1.0


def test_multi_trend_no_lookahead():
    prices = pd.Series(np.arange(1, 11, dtype=float), index=_idx(10))
    sig = multi_trend_signal(prices, lookbacks=(2, 4))
    mutated = prices.copy()
    mutated.iloc[-1] = 999.0
    sig_mut = multi_trend_signal(mutated, lookbacks=(2, 4))
    assert sig.iloc[5] == sig_mut.iloc[5]


def test_rebalance_band_holds_small_changes():
    target = pd.Series([0.0, 0.3, 0.35, 0.7], index=_idx(4))
    out = rebalance_band(target, band=0.1)
    # 0 (no move) -> adopt 0.3 -> hold (0.05<band) -> adopt 0.7
    assert list(np.round(out.values, 6)) == [0.0, 0.3, 0.3, 0.7]


def test_rebalance_band_reduces_number_of_changes():
    rng = np.random.default_rng(0)
    noisy = pd.Series(np.clip(0.5 + rng.normal(0, 0.05, 200), 0, 1), index=_idx(200))
    banded = rebalance_band(noisy, band=0.1)
    raw_changes = int((noisy.diff().abs() > 1e-9).sum())
    banded_changes = int((banded.diff().abs() > 1e-9).sum())
    assert banded_changes < raw_changes
