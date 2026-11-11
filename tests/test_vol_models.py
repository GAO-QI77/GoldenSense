"""Tests for the HAR-RV volatility layer and distribution bands."""
import numpy as np
import pandas as pd
import pytest

from vol_models import (
    HARRVModel,
    band_coverage,
    forecast_return_bands,
    realized_variance_components,
)


def _synthetic_prices(n: int = 1500, seed: int = 7) -> pd.Series:
    """Two-regime vol process so vol is genuinely forecastable."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 120) % 2 == 0, 0.008, 0.020)
    rets = rng.normal(0.0003, vol)
    prices = 2000.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="Gold")


def test_har_components_are_causal():
    prices = _synthetic_prices()
    rets = prices.pct_change().dropna()
    comps = realized_variance_components(rets)
    # Component at date t must not change when future data is appended.
    partial = realized_variance_components(rets.iloc[:800])
    pd.testing.assert_series_equal(
        comps["rv_m"].iloc[:800].dropna(), partial["rv_m"].dropna()
    )


def test_har_forecast_tracks_regime_shift():
    prices = _synthetic_prices()
    rets = prices.pct_change().dropna()
    model = HARRVModel(horizon=5).fit(rets)
    preds = model.predict_series_ann_vol(rets)
    # Average forecast during calm blocks must be lower than in stormy blocks.
    daily_vol = rets.rolling(20).std().reindex(preds.index)
    calm = preds[daily_vol < daily_vol.median()]
    stormy = preds[daily_vol >= daily_vol.median()]
    assert calm.mean() < stormy.mean()
    assert (preds > 0).all()


def test_har_beats_unconditional_vol_forecast():
    prices = _synthetic_prices()
    rets = prices.pct_change().dropna()
    model = HARRVModel(horizon=1).fit(rets)
    preds_var = (model.predict_series_ann_vol(rets) ** 2) / 252.0
    realized = (rets ** 2).shift(-1).reindex(preds_var.index)
    mask = realized.notna()
    har_mse = float(((preds_var - realized)[mask] ** 2).mean())
    naive_mse = float(((realized[mask].mean() - realized[mask]) ** 2).mean())
    assert har_mse < naive_mse


def test_forecast_return_bands_ordering_and_fields():
    prices = _synthetic_prices()
    bands = forecast_return_bands(prices, horizon_days=5)
    assert bands["p10"] < bands["p50"] < bands["p90"]
    assert bands["ann_vol_forecast"] > 0
    assert bands["horizon_days"] == 5


def test_forecast_return_bands_rejects_short_history():
    prices = _synthetic_prices(n=100)
    with pytest.raises(ValueError):
        forecast_return_bands(prices, horizon_days=5)


def test_band_coverage_close_to_nominal():
    prices = _synthetic_prices(n=2000)
    cov = band_coverage(prices, horizon_days=1, lo_q=0.10, hi_q=0.90)
    # Nominal coverage is 0.80; allow sampling slack on synthetic data.
    assert 0.70 <= cov <= 0.90


def test_21_session_coverage_calibrates_only_with_mature_returns():
    prices = _synthetic_prices(n=345)
    horizon = 21
    volatility = prices.pct_change().rolling(22).std() * np.sqrt(horizon)
    standardized = (prices.shift(-horizon) / prices - 1.0) / volatility
    hits = []
    for i in range(horizon, len(prices) - horizon):
        matured = standardized.iloc[: i - horizon + 1].dropna()
        if len(matured) >= 252:
            low, high = np.quantile(matured, [0.10, 0.90])
            hits.append(low <= standardized.iloc[i] <= high)

    assert band_coverage(prices, horizon_days=horizon) == pytest.approx(np.mean(hits))


def test_short_term_comparison_scores_only_mature_labels_and_is_causal():
    from vol_models import walk_forward_variance
    rng = np.random.default_rng(43)
    dates = pd.bdate_range("2024-01-01", periods=200)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates))), index=dates)
    original = walk_forward_variance(prices, horizon=5, min_history=100)
    changed = prices.copy()
    changed.iloc[150:] *= 2
    alternative = walk_forward_variance(changed, horizon=5, min_history=100)

    assert original.index[-1] == dates[-6]
    assert original["matures_on"].iloc[-1] == dates[-1]
    forecasts = ["rolling_variance", "ewma_variance", "har_variance"]
    pd.testing.assert_frame_equal(
        original.loc[:dates[149], forecasts], alternative.loc[:dates[149], forecasts]
    )
    expected = (prices.pct_change().loc[dates[150]:dates[154]] ** 2).sum()
    assert original.loc[dates[149], "realized_variance"] == pytest.approx(expected)
