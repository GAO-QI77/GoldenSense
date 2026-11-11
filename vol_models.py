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
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    prices = prices.astype(float).dropna().sort_index()
    returns = prices.pct_change().dropna()
    trail = trailing_ann_vol(returns, window=window) * np.sqrt(horizon_days / TRADING_DAYS)
    fwd = prices.shift(-horizon_days) / prices - 1.0
    z = (fwd / trail).replace([np.inf, -np.inf], np.nan)

    hits = []
    values = z.dropna()
    # Measure maturity on the original session axis, even if some z-scores
    # were excluded because their trailing volatility was unavailable.
    starts = np.flatnonzero(z.notna())
    for i, decision_pos in enumerate(starts):
        eligible = starts.searchsorted(decision_pos - horizon_days, side="right")
        if eligible < 252:
            continue
        past = values.iloc[:eligible]
        lo, hi = np.quantile(past.values, lo_q), np.quantile(past.values, hi_q)
        realized = values.iloc[i]
        hits.append(lo <= realized <= hi)
    if not hits:
        raise ValueError("not enough data for coverage check")
    return float(np.mean(hits))


def walk_forward_variance(
    prices: pd.Series, *, horizon: int, min_history: int = 260,
    include_challengers: bool = False,
) -> pd.DataFrame:
    """Evaluate fixed rolling, EWMA and HAR baselines on identical mature origins.

    Inputs are one instrument's ordered, completed daily closes. The target is
    the sum of squared simple returns over the next `horizon` observations.
    The training sample expands through each origin, with HAR labels ending
    no later than that origin. EWMA uses fixed decay 0.94.
    """
    if horizon < 1 or min_history < 81 + horizon:
        raise ValueError("need a positive horizon and at least 81 + horizon training returns")
    if not isinstance(prices.index, pd.DatetimeIndex) or (
        prices.index.hasnans or not prices.index.is_unique or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("prices must have unique, increasing timestamps")
    prices = prices.astype(float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("prices must be finite and positive")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if len(returns) < min_history + horizon:
        raise ValueError("insufficient history for a mature out-of-sample forecast")

    squared = returns.pow(2)
    rolling = squared.rolling(22).mean()
    ewma = squared.ewm(alpha=0.06, adjust=False).mean()
    rows = []
    if include_challengers:
        log_design = np.log(HARRVModel(horizon=horizon)._design(returns).clip(lower=1e-12)).to_numpy()
        log_x = np.column_stack((np.ones(len(log_design)), log_design[:, :3]))
        log_target = log_design[:, 3]
        ratio = 1.0
        matured_count = 0
    for i in range(min_history - 1, len(returns) - horizon):
        history = returns.iloc[:i + 1]
        har = HARRVModel(horizon=horizon).fit(history)
        har_daily = har.predict_latest_ann_vol(history) ** 2 / TRADING_DAYS
        row = {
            "decision_date": returns.index[i],
            "matures_on": returns.index[i + horizon],
            "realized_variance": float(squared.iloc[i + 1:i + 1 + horizon].sum()),
            "rolling_variance": max(float(rolling.iloc[i]), 1e-12) * horizon,
            "ewma_variance": max(float(ewma.iloc[i]), 1e-12) * horizon,
            "har_variance": har_daily * horizon,
        }
        if include_challengers:
            # Labels at origins after i-horizon have not matured.
            train = slice(21, i - horizon + 1)
            x, y = log_x[train], log_target[train]
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            fitted = np.einsum("ij,j->i", x, coef)
            # Convert the log fit back to the conditional arithmetic variance.
            smearing = np.exp(np.clip(y - fitted, -20, 20)).mean()
            daily = np.exp(np.clip(np.dot(log_x[i], coef), -28, 5)) * smearing
            forecast = float(max(daily, 1e-12) * horizon)
            if len(rows) >= horizon:
                matured = rows[-horizon]
                observed = matured["realized_variance"] / matured["log_har_variance"]
                ratio = 0.94 * ratio + 0.06 * float(np.clip(observed, 0.25, 4.0))
                matured_count += 1
            multiplier = float(np.clip(ratio, 0.5, 2.0)) if matured_count >= 30 else 1.0
            row.update(log_har_variance=forecast, log_har_vra_variance=forecast * multiplier)
        rows.append(row)
    return pd.DataFrame.from_records(rows, index="decision_date")
