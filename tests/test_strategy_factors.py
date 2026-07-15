"""Tests for rules-based factor signal primitives.

The one correctness property that matters for an honest backtest: a signal at
time t must depend only on data available at or before t (no look-ahead). These
tests pin that down plus the basic semantics of each primitive.
"""
import numpy as np
import pandas as pd
import pytest

from strategy_factors import (
    forward_returns_1d,
    momentum_signal,
    trend_signal,
    vol_target_position,
)


def _idx(n):
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_forward_returns_align_to_decision_date():
    prices = pd.Series([100.0, 110.0, 99.0], index=_idx(3))
    fwd = forward_returns_1d(prices)
    # fwd[t] = price[t+1]/price[t]-1; last date has no future -> dropped.
    assert list(np.round(fwd.values, 6)) == [0.10, -0.10]
    assert len(fwd) == 2


def test_trend_signal_long_above_ma_flat_below():
    rising = pd.Series(np.arange(1, 11, dtype=float), index=_idx(10))
    sig = trend_signal(rising, lookback=3)
    # While rising, price is above its trailing mean -> long (1).
    assert sig.iloc[-1] == 1.0
    falling = pd.Series(np.arange(10, 0, -1, dtype=float), index=_idx(10))
    assert trend_signal(falling, lookback=3).iloc[-1] == 0.0


def test_trend_signal_has_no_lookahead():
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=_idx(5))
    sig = trend_signal(prices, lookback=2)
    # Mutating a FUTURE price must not change an earlier signal value.
    mutated = prices.copy()
    mutated.iloc[4] = 999.0
    sig_mut = trend_signal(mutated, lookback=2)
    assert sig.iloc[2] == sig_mut.iloc[2]
    assert sig.iloc[3] == sig_mut.iloc[3]


def test_momentum_signal_sign():
    up = pd.Series([100.0, 101.0, 102.0, 105.0], index=_idx(4))
    assert momentum_signal(up, lookback=2).iloc[-1] == 1.0
    down = pd.Series([100.0, 99.0, 98.0, 95.0], index=_idx(4))
    assert momentum_signal(down, lookback=2).iloc[-1] == -1.0


def test_vol_target_scales_inversely_with_realized_vol():
    # Low-vol then high-vol regime; same base long position.
    rets = pd.Series(
        [0.001, -0.001, 0.001, -0.001, 0.05, -0.05, 0.05, -0.05], index=_idx(8)
    )
    base = pd.Series(1.0, index=_idx(8))
    pos = vol_target_position(
        base, rets, target_vol=0.10, lookback=3, max_leverage=3.0
    )
    # Position in the calm window should exceed position in the stormy window.
    assert pos.iloc[3] > pos.iloc[-1]
    # Never exceeds the leverage cap.
    assert pos.max() <= 3.0 + 1e-9


def test_vol_target_respects_zero_base():
    rets = pd.Series([0.01, -0.01, 0.02, -0.02], index=_idx(4))
    base = pd.Series([0.0, 0.0, 0.0, 0.0], index=_idx(4))
    pos = vol_target_position(base, rets, target_vol=0.1, lookback=2, max_leverage=3.0)
    assert (pos.values == 0.0).all()
