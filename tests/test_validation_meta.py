"""Tests for purged walk-forward, Deflated Sharpe, and meta-labeling."""
import numpy as np
import pandas as pd
import pytest

from meta_labeling import (
    meta_filtered_positions,
    state_features,
    triple_barrier_labels,
)
from validation import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    purged_walk_forward_splits,
    sharpe_report,
)


# ------------------------------ validation -------------------------------- #
def test_purged_splits_respect_embargo_and_do_not_overlap():
    splits = list(purged_walk_forward_splits(500, min_train=200, test_size=50, embargo=10))
    assert splits
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()
        # Embargo gap: at least 10 indices between train end and test start.
        assert test_idx.min() - train_idx.max() > 10
    covered = np.concatenate([t for _, t in splits])
    assert len(np.unique(covered)) == len(covered)  # test blocks never overlap
    assert covered.max() == 499


def test_purged_splits_raise_when_too_small():
    with pytest.raises(ValueError):
        list(purged_walk_forward_splits(50, min_train=100, test_size=10))


def test_psr_increases_with_sharpe_and_sample():
    lo = probabilistic_sharpe_ratio(0.02, benchmark_sr=0.0, n_obs=252)
    hi = probabilistic_sharpe_ratio(0.10, benchmark_sr=0.0, n_obs=252)
    assert hi > lo
    longer = probabilistic_sharpe_ratio(0.05, benchmark_sr=0.0, n_obs=2520)
    shorter = probabilistic_sharpe_ratio(0.05, benchmark_sr=0.0, n_obs=252)
    assert longer > shorter


def test_expected_max_sharpe_grows_with_trials():
    v = 1.0 / 251
    assert expected_max_sharpe(1, v) == 0.0
    assert expected_max_sharpe(100, v) > expected_max_sharpe(10, v) > 0.0


def test_dsr_penalizes_multiple_testing():
    few = deflated_sharpe_ratio(0.08, n_trials=2, n_obs=1000)
    many = deflated_sharpe_ratio(0.08, n_trials=200, n_obs=1000)
    assert few > many


def test_sharpe_report_fields():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, 1000)
    rep = sharpe_report(rets, n_trials=5)
    assert set(rep) >= {"sharpe_ann", "psr", "dsr", "n_obs"}
    assert 0.0 <= rep["psr"] <= 1.0
    assert 0.0 <= rep["dsr"] <= 1.0
    assert rep["dsr"] <= rep["psr"]  # deflation can only reduce confidence


# ----------------------------- meta-labeling ------------------------------ #
def _price_path() -> pd.Series:
    idx = pd.date_range("2023-01-01", periods=80, freq="B")
    base = np.full(80, 100.0)
    # Warm-up noise so trailing vol is defined and positive.
    base[:40] += np.sin(np.arange(40)) * 0.8
    # After the entry at i=40: steady rally -> profit target should hit.
    base[40:] = 100 + np.arange(40) * 1.5
    return pd.Series(base, index=idx)


def test_triple_barrier_profit_target_hit():
    prices = _price_path()
    entries = pd.Series(0.0, index=prices.index)
    entries.iloc[40] = 1.0
    labels = triple_barrier_labels(prices, entries, pt_mult=1.0, sl_mult=1.0, max_holding_days=10)
    assert len(labels) == 1
    assert labels.iloc[0] == 1.0


def test_triple_barrier_stop_loss_hit():
    prices = _price_path()
    prices.iloc[41:] = 100 - np.arange(39) * 1.5  # crash after entry
    entries = pd.Series(0.0, index=prices.index)
    entries.iloc[40] = 1.0
    labels = triple_barrier_labels(prices, entries, pt_mult=1.0, sl_mult=1.0, max_holding_days=10)
    assert labels.iloc[0] == 0.0


def test_meta_filter_only_acts_where_it_has_an_opinion():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    base = pd.Series([1.0, 1.0, 1.0, 0.0, 1.0, 1.0], index=idx)
    probs = pd.Series({idx[1]: 0.2, idx[4]: 0.9})
    out = meta_filtered_positions(base, probs, threshold=0.5)
    assert out.iloc[0] == 1.0  # no opinion -> keep primary
    assert out.iloc[1] == 0.0  # low prob -> filtered out
    assert out.iloc[4] == 1.0  # high prob -> kept


def test_state_features_are_causal_and_cover_macro_columns():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2022-01-01", periods=300, freq="B")
    raw = pd.DataFrame(
        {
            "Gold": 2000 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))),
            "Real_10Y": 1.5 + np.cumsum(rng.normal(0, 0.02, 300)),
            "USD_Index": 100 + np.cumsum(rng.normal(0, 0.2, 300)),
            "VIX": np.abs(rng.normal(18, 4, 300)),
        },
        index=idx,
    )
    feats = state_features(raw)
    assert {"vol_22d", "trend_200", "real_rate_chg_63d", "usd_chg_63d", "vix"} <= set(feats.columns)
    # Causality: features at date t identical when future rows are dropped.
    partial = state_features(raw.iloc[:200])
    pd.testing.assert_frame_equal(feats.iloc[:200].dropna(), partial.dropna())
