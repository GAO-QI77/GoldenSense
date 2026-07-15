"""Tests for the flagship integrated strategy (causal, drawdown-aware)."""
import numpy as np
import pandas as pd
import pytest

from regime_probabilistic import GaussianHMM, causal_regime_stress, regime_features
from strategy_integrated import (
    build_flagship_positions,
    evaluate_flagship,
)


def _macro_frame(n: int = 1400, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    vol = np.where((np.arange(n) // 130) % 2 == 0, 0.007, 0.02)
    gold = pd.Series(1500 * np.exp(np.cumsum(rng.normal(0.0004, vol))), index=idx)
    return pd.DataFrame(
        {
            "Gold": gold,
            "USD_Index": 100 + np.cumsum(rng.normal(0, 0.2, n)),
            "Real_10Y": 1.5 + np.cumsum(rng.normal(0, 0.02, n)),
            "Breakeven_10Y": 2.2 + np.cumsum(rng.normal(0, 0.01, n)),
        },
        index=idx,
    )


# ------------------------- causal HMM primitives -------------------------- #
def test_filter_posterior_is_causal():
    prices = _macro_frame()["Gold"]
    feats = regime_features(prices)
    model = GaussianHMM(n_states=3, max_iter=30).fit(feats.values)
    full = model.filter_posterior(feats.values)
    # Filtered value at t must not change when future rows are withheld.
    partial = model.filter_posterior(feats.values[:600])
    assert np.allclose(full[:600], partial, atol=1e-8)
    assert np.allclose(full.sum(axis=1), 1.0, atol=1e-6)


def test_causal_regime_stress_no_lookahead():
    prices = _macro_frame(n=1400)["Gold"]
    stress = causal_regime_stress(prices, min_train=400, refit_every=120)
    assert stress is not None
    assert stress.between(0.0, 1.0).all()
    # Recomputing on a truncated series must reproduce the earlier values
    # exactly (walk-forward refit is deterministic and past-only).
    truncated = causal_regime_stress(prices.iloc[:900], min_train=400, refit_every=120)
    common = stress.index.intersection(truncated.index)
    assert len(common) > 100
    assert np.allclose(stress.loc[common].values, truncated.loc[common].values, atol=1e-6)


def test_causal_regime_stress_short_history_returns_none():
    prices = _macro_frame(n=300)["Gold"]
    assert causal_regime_stress(prices, min_train=756) is None


# ----------------------------- flagship ----------------------------------- #
def test_flagship_positions_are_bounded_and_aligned():
    raw = _macro_frame()
    pos = build_flagship_positions(raw)
    assert pos.index.equals(raw.index)
    assert (pos >= 0.0).all()
    assert (pos <= 2.0 + 1e-9).all()  # vol-target max_leverage cap


def test_flagship_degrades_without_macro_columns():
    raw = _macro_frame()[["Gold"]]
    pos = build_flagship_positions(raw)  # no Real_10Y -> rate scale = 1
    assert pos.index.equals(raw.index)
    assert (pos >= 0.0).all()


def test_evaluate_flagship_returns_curves_and_metrics():
    raw = _macro_frame(n=1400)
    result = evaluate_flagship(raw)
    assert result is not None
    for key in ("sharpe", "sortino", "max_drawdown", "calmar", "dsr"):
        assert key in result.metrics
        assert key in result.benchmark
    # Curves are downsampled and aligned across the two series.
    assert len(result.curve_dates) == len(result.flagship_equity) == len(result.benchmark_equity)
    assert len(result.curve_dates) <= 221
    assert result.flagship_equity[0] > 0


def test_evaluate_flagship_dsr_in_unit_interval():
    result = evaluate_flagship(_macro_frame(n=1400))
    assert 0.0 <= result.metrics["dsr"] <= 1.0
    assert result.metrics["dsr"] <= result.metrics["psr"] + 1e-9


def test_evaluate_flagship_short_sample_returns_none():
    raw = _macro_frame(n=300)
    assert evaluate_flagship(raw) is None
