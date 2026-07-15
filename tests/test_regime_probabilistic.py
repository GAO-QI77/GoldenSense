"""Tests for the numpy Gaussian HMM regime model and the blended decision."""
import numpy as np
import pandas as pd

from regime_probabilistic import (
    GaussianHMM,
    blended_exposure,
    fit_gold_regime_posterior,
    regime_features,
)
from regime_strategy import evaluate_regime, evaluate_regime_v2


def _two_regime_prices(n: int = 1200, seed: int = 3) -> pd.Series:
    """Calm first half, stressed second half."""
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(n) < n // 2, 0.006, 0.025)
    rets = rng.normal(0.0004, vol)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(2000.0 * np.exp(np.cumsum(rets)), index=idx)


def test_hmm_separates_volatility_regimes():
    prices = _two_regime_prices()
    feats = regime_features(prices)
    model = GaussianHMM(n_states=2, max_iter=40).fit(feats.values)
    gamma = model.posterior(feats.values)

    n = len(gamma)
    # State 0 is relabelled to lowest vol: it should dominate the calm half.
    calm_prob_first = gamma[: n // 2 - 20, 0].mean()
    calm_prob_second = gamma[n // 2 + 20 :, 0].mean()
    assert calm_prob_first > 0.7
    assert calm_prob_second < 0.3


def test_hmm_posterior_rows_sum_to_one():
    prices = _two_regime_prices()
    feats = regime_features(prices)
    model = GaussianHMM(n_states=3, max_iter=30).fit(feats.values)
    gamma = model.posterior(feats.values)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-6)
    assert gamma.min() >= 0.0


def test_fit_gold_regime_posterior_short_history_returns_none():
    prices = _two_regime_prices(n=120)
    assert fit_gold_regime_posterior(prices) is None


def test_fit_gold_regime_posterior_labels():
    prices = _two_regime_prices()
    post = fit_gold_regime_posterior(prices, n_states=3)
    assert list(post.columns) == ["calm", "elevated", "stress"]
    assert len(post) > 1000


def test_blended_exposure_probability_weighting():
    exposures = {"calm": 1.0, "elevated": 0.7, "stress": 0.4}
    assert blended_exposure({"calm": 1.0}, exposures) == 1.0
    mixed = blended_exposure({"calm": 0.5, "stress": 0.5}, exposures)
    assert abs(mixed - 0.7) < 1e-9


def test_evaluate_regime_v2_falls_back_without_posterior():
    prices = _two_regime_prices()
    v1 = evaluate_regime(prices, risk_profile="balanced")
    v2 = evaluate_regime_v2(prices, risk_profile="balanced", state_posterior=None)
    assert v2.regime_model == "rule_threshold"
    assert v2.target_exposure_pct == v1.target_exposure_pct


def test_evaluate_regime_v2_blends_posterior():
    prices = _two_regime_prices()
    decision = evaluate_regime_v2(
        prices,
        risk_profile="balanced",
        state_posterior={"calm": 0.6, "elevated": 0.3, "stress": 0.1},
    )
    assert decision.regime_model == "hmm_posterior_blend"
    assert decision.state_probabilities is not None
    assert abs(sum(decision.state_probabilities.values()) - 1.0) < 1e-3
    # Blended scale = .6*1 + .3*.7 + .1*.4 = 0.85 -> between calm and elevated caps.
    rule_calm = evaluate_regime_v2(
        prices, risk_profile="balanced", state_posterior={"calm": 1.0}
    )
    rule_stress = evaluate_regime_v2(
        prices, risk_profile="balanced", state_posterior={"stress": 1.0}
    )
    if decision.regime != "downtrend":
        assert rule_stress.target_exposure_pct <= decision.target_exposure_pct <= rule_calm.target_exposure_pct


def test_evaluate_regime_v2_stress_posterior_lowers_confidence():
    prices = _two_regime_prices()
    decision = evaluate_regime_v2(
        prices,
        risk_profile="balanced",
        state_posterior={"calm": 0.1, "elevated": 0.2, "stress": 0.7},
    )
    assert decision.confidence_band == "低"
    assert decision.vol_state == "stress"
