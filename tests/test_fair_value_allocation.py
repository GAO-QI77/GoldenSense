"""Tests for the fair-value anchor and the allocation/scenario layer."""
import numpy as np
import pandas as pd
import pytest

from allocation import (
    allocation_range,
    monte_carlo_cone,
    regime_tilt_from_posterior,
    valuation_tilt_from_deviation,
)
from fair_value import fit_fair_value


def _macro_frame(n: int = 1500, seed: int = 5) -> pd.DataFrame:
    """Gold cointegrated with real yield + USD so the anchor is recoverable."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n, freq="B")
    real = 1.0 + np.cumsum(rng.normal(0, 0.01, n))
    log_usd = np.log(100) + np.cumsum(rng.normal(0, 0.001, n))
    noise = np.cumsum(rng.normal(0, 0.004, n)) * 0.1
    log_gold = 8.5 - 0.25 * real - 0.8 * (log_usd - np.log(100)) + noise
    return pd.DataFrame(
        {"Gold": np.exp(log_gold), "Real_10Y": real, "USD_Index": np.exp(log_usd)},
        index=idx,
    )


def test_fair_value_recovers_negative_real_rate_beta():
    result = fit_fair_value(_macro_frame())
    assert result is not None
    assert result.coefficients["real_10y"] < 0  # higher real yield -> cheaper gold
    assert result.r_squared > 0.5
    assert result.n_obs > 1000
    assert abs(result.deviation_pct) < 50


def test_fair_value_returns_none_without_macro_columns():
    frame = _macro_frame()[["Gold"]]
    assert fit_fair_value(frame) is None


def test_fair_value_returns_none_on_short_sample():
    assert fit_fair_value(_macro_frame(n=200)) is None


def test_fair_value_quartile_card_present():
    result = fit_fair_value(_macro_frame())
    assert "q1_cheapest" in result.quartile_forward_returns
    assert "q4_richest" in result.quartile_forward_returns


def test_fair_value_reports_deviation_band_and_interpretation():
    result = fit_fair_value(_macro_frame())
    payload = result.as_dict()
    # New framing fields must always be present and self-consistent.
    for key in ("deviation_z", "band_std_pct", "regime_break", "interpretation"):
        assert key in payload
    assert payload["band_std_pct"] >= 0
    assert isinstance(payload["regime_break"], bool)
    assert isinstance(payload["interpretation"], str) and payload["interpretation"]


def test_fair_value_flags_structural_break_on_extreme_deviation():
    # Push the latest gold print far above the cointegrating relationship so
    # the current residual sits well beyond its historical band.
    frame = _macro_frame()
    frame.iloc[-1, frame.columns.get_loc("Gold")] *= 2.2
    result = fit_fair_value(frame)
    assert abs(result.deviation_z) >= 2.0
    assert result.regime_break is True
    assert "结构性偏离期" in result.interpretation
    # A structural-break reading must NOT be phrased as a clean over/undervaluation.
    assert "偏贵" not in result.interpretation


def test_fair_value_normal_deviation_is_not_flagged():
    result = fit_fair_value(_macro_frame())
    # The synthetic frame is cointegrated by construction -> small deviation.
    if abs(result.deviation_z) < 1.0:
        assert result.regime_break is False
        assert "结构性偏离期" not in result.interpretation


def test_regime_tilt_sign_and_bounds():
    assert regime_tilt_from_posterior({"calm": 1.0}) == pytest.approx(0.25)
    assert regime_tilt_from_posterior({"stress": 1.0}) == pytest.approx(-0.25)
    assert regime_tilt_from_posterior({"elevated": 1.0}) == 0.0
    assert regime_tilt_from_posterior(None) == 0.0


def test_valuation_tilt_direction_and_saturation():
    assert valuation_tilt_from_deviation(-30.0) == pytest.approx(0.25)  # cheap -> up
    assert valuation_tilt_from_deviation(60.0) == pytest.approx(-0.25)  # rich -> down, capped
    assert valuation_tilt_from_deviation(None) == 0.0


def test_allocation_range_tilts_prior_not_replaces():
    calm_cheap = allocation_range(
        "balanced",
        regime_posterior={"calm": 1.0},
        valuation_deviation_pct=-30.0,
    )
    stress_rich = allocation_range(
        "balanced",
        regime_posterior={"stress": 1.0},
        valuation_deviation_pct=30.0,
    )
    neutral = allocation_range("balanced")

    assert calm_cheap.recommended_range_pct[1] > neutral.recommended_range_pct[1]
    assert stress_rich.recommended_range_pct[1] < neutral.recommended_range_pct[1]
    # Combined multiplier is clipped to [0.5, 1.5] of the prior.
    lo, hi = neutral.prior_range_pct
    assert stress_rich.recommended_range_pct[0] >= lo * 0.5 - 1e-9
    assert calm_cheap.recommended_range_pct[1] <= hi * 1.5 + 1e-9


def test_allocation_unknown_profile_falls_back_to_balanced():
    advice = allocation_range("degen")
    assert advice.profile == "balanced"


def _regime_prices(n: int = 1200, seed: int = 9) -> pd.Series:
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 150) % 2 == 0, 0.007, 0.02)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(2000 * np.exp(np.cumsum(rng.normal(0.0004, vol))), index=idx)


def test_monte_carlo_cone_shape_and_ordering():
    cone = monte_carlo_cone(_regime_prices(), horizon_days=60, n_paths=500, checkpoints=(30, 60))
    assert cone is not None
    assert len(cone["percentile_paths"]["p50"]) == 60
    d30 = cone["checkpoints"]["d30"]
    assert d30["p10"] < d30["p50"] < d30["p90"]
    assert 0.0 <= d30["prob_above_spot"] <= 1.0
    # Cone must widen with horizon.
    d60 = cone["checkpoints"]["d60"]
    assert (d60["p90"] - d60["p10"]) > (d30["p90"] - d30["p10"])


def test_monte_carlo_cone_is_deterministic_given_seed():
    a = monte_carlo_cone(_regime_prices(), horizon_days=30, n_paths=300, checkpoints=(30,))
    b = monte_carlo_cone(_regime_prices(), horizon_days=30, n_paths=300, checkpoints=(30,))
    assert a["checkpoints"]["d30"]["p50"] == b["checkpoints"]["d30"]["p50"]


def test_monte_carlo_cone_returns_none_on_short_history():
    assert monte_carlo_cone(_regime_prices(n=120)) is None
