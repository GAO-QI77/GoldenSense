"""Short-horizon volatility and return-distribution models.

Stage-A showed daily *direction* has no out-of-sample edge, so the short-term
layer predicts what actually is predictable:

- ``HARRVModel``      -- Corsi (2009) heterogeneous autoregression on realized
                         variance (daily / weekly / monthly components). With
                         daily closes the daily RV proxy is the squared return;
                         the weekly and monthly aggregates smooth its noise.
- ``forecast_return_bands`` -- turns the vol forecast into honest P10/P50/P90
                         return bands via empirical quantiles of vol-scaled
                         (standardized) returns. No Gaussian assumption.

All computations are causal: a forecast for date t uses data up to and
including t only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realized_variance_components(returns: pd.Series) -> pd.DataFrame:
    """Daily / weekly / monthly realized-variance components (HAR inputs)."""
    rv_d = returns.astype(float) ** 2
    return pd.DataFrame(
        {
            "rv_d": rv_d,
            "rv_w": rv_d.rolling(5).mean(),
            "rv_m": rv_d.rolling(22).mean(),
        }
    )


@dataclass
class HARFit:
    intercept: float
    beta_d: float
    beta_w: float
    beta_m: float
    horizon: int
    n_obs: int


class HARRVModel:
    """HAR-RV via OLS. Target = mean realized variance over the next ``horizon`` days."""

    def __init__(self, horizon: int = 1):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = int(horizon)
        self.fit_: Optional[HARFit] = None

    def _design(self, returns: pd.Series) -> pd.DataFrame:
        comps = realized_variance_components(returns)
        rv_d = comps["rv_d"]
        # Forward mean RV over the next `horizon` days (the prediction target).
        fwd = rv_d.shift(-1).rolling(self.horizon).mean().shift(-(self.horizon - 1))
        comps["target"] = fwd
        return comps

    def fit(self, returns: pd.Series) -> "HARRVModel":
        frame = self._design(returns).dropna()
        if len(frame) < 60:
            raise ValueError(f"need >= 60 observations to fit HAR, got {len(frame)}")
        X = np.column_stack(
            [np.ones(len(frame)), frame["rv_d"], frame["rv_w"], frame["rv_m"]]
        )
        y = frame["target"].values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.fit_ = HARFit(
            intercept=float(coef[0]),
            beta_d=float(coef[1]),
            beta_w=float(coef[2]),
            beta_m=float(coef[3]),
            horizon=self.horizon,
            n_obs=len(frame),
        )
        return self

    def _predict_from_components(self, rv_d: float, rv_w: float, rv_m: float) -> float:
        f = self.fit_
        if f is None:
            raise RuntimeError("model is not fitted")
        pred = f.intercept + f.beta_d * rv_d + f.beta_w * rv_w + f.beta_m * rv_m
        # Realized variance cannot be negative; floor at a tiny epsilon.
        return float(max(pred, 1e-12))

    def predict_latest_ann_vol(self, returns: pd.Series) -> float:
        """Annualized vol forecast for the next ``horizon`` days, using data up to now."""
        comps = realized_variance_components(returns).dropna()
        if comps.empty:
            raise ValueError("not enough data for HAR components")
        last = comps.iloc[-1]
        var_daily = self._predict_from_components(last["rv_d"], last["rv_w"], last["rv_m"])
        return float(np.sqrt(var_daily * TRADING_DAYS))

    def predict_series_ann_vol(self, returns: pd.Series) -> pd.Series:
        """Causal per-date vol forecasts (each uses components known at that date)."""
        comps = realized_variance_components(returns).dropna()
        preds = [
            self._predict_from_components(row.rv_d, row.rv_w, row.rv_m)
            for row in comps.itertuples()
        ]
        return pd.Series(np.sqrt(np.array(preds) * TRADING_DAYS), index=comps.index)


def trailing_ann_vol(returns: pd.Series, window: int = 22) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def forecast_return_bands(
    prices: pd.Series,
    *,
    horizon_days: int,
    quantiles: tuple = (0.10, 0.50, 0.90),
    min_history: int = 260,
) -> Dict[str, float]:
    """Distributional forecast of the ``horizon_days`` forward return.

    Method: fit HAR on the full history, forecast the next-period vol, then
    scale empirical quantiles of *standardized* historical h-day returns by
    that forecast. Standardizing by trailing vol before taking quantiles keeps
    fat tails and skew from the data instead of assuming a Gaussian.
    """
    prices = prices.astype(float).dropna()
    returns = prices.pct_change().dropna()
    if len(returns) < min_history:
        raise ValueError(f"need >= {min_history} return observations, got {len(returns)}")

    model = HARRVModel(horizon=min(horizon_days, 22)).fit(returns)
    ann_vol = model.predict_latest_ann_vol(returns)
    horizon_vol = ann_vol * np.sqrt(horizon_days / TRADING_DAYS)

    # Standardized h-day forward returns over history (causal scaling at each t).
    trail = trailing_ann_vol(returns, window=22) * np.sqrt(horizon_days / TRADING_DAYS)
    fwd_h = prices.shift(-horizon_days) / prices - 1.0
    z = (fwd_h / trail).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 60:
        raise ValueError("not enough standardized samples for quantiles")

    bands = {f"p{int(q * 100)}": float(np.quantile(z.values, q) * horizon_vol) for q in quantiles}
    bands["ann_vol_forecast"] = float(ann_vol)
    bands["horizon_days"] = float(horizon_days)
    return bands


def band_coverage(
    prices: pd.Series,
    *,
    horizon_days: int,
    lo_q: float = 0.10,
    hi_q: float = 0.90,
    window: int = 22,
) -> float:
    """Fraction of realized h-day returns inside the [lo_q, hi_q] causal band.

    A calibration diagnostic: for honest bands this should sit near
    ``hi_q - lo_q``. Uses trailing-vol-scaled empirical quantiles per date.
    """
    prices = prices.astype(float).dropna()
    returns = prices.pct_change().dropna()
    trail = trailing_ann_vol(returns, window=window) * np.sqrt(horizon_days / TRADING_DAYS)
    fwd = prices.shift(-horizon_days) / prices - 1.0
    z = (fwd / trail).replace([np.inf, -np.inf], np.nan)

    hits = []
    values = z.dropna()
    # Expanding, strictly-causal quantile estimates (first 252 obs warm-up).
    for i in range(252, len(values)):
        past = values.iloc[:i]
        lo, hi = np.quantile(past.values, lo_q), np.quantile(past.values, hi_q)
        realized = values.iloc[i]
        hits.append(lo <= realized <= hi)
    if not hits:
        raise ValueError("not enough data for coverage check")
    return float(np.mean(hits))
