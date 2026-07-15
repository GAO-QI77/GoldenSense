"""Long-horizon fair-value anchor for gold (error-correction framing).

The long-term layer does not predict prices. It answers two questions a
research terminal can defend:

1. Where does gold trade *relative to its macro fair value*? Fair value is a
   cointegrating-style levels regression fitted on the long sample:

       log(gold) ~ a + b * real_10y + c * log(usd_index)

   The residual is the misvaluation gauge (in %). Real 10Y TIPS yield is the
   canonical anchor; USD adds the denomination effect.

2. Historically, what happened *after* similar misvaluation levels? We report
   mean forward returns by deviation quartile -- an evidence card, not a
   signal.

An error-correction regression Δlog(gold)_t = λ * resid_{t-1} + ε estimates
how fast deviations decay; a negative λ with a sane half-life is what makes
the anchor meaningful rather than decorative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class FairValueResult:
    deviation_pct: float                 # +12.0 means 12% above macro fair value
    fair_value: float                    # model fair price (USD)
    spot: float                          # latest observed price
    half_life_days: Optional[float]      # error-correction half-life; None if λ>=0
    lambda_daily: float                  # error-correction speed (per day)
    r_squared: float                     # levels-regression fit quality
    deviation_z: float = 0.0             # current deviation in std of its own history
    band_std_pct: float = 0.0            # 1 std of the deviation series, in % points
    regime_break: bool = False           # deviation beyond normal band -> caveat
    interpretation: str = ""             # honest, caveated reading (not a signal)
    coefficients: Dict[str, float] = field(default_factory=dict)
    deviation_series_tail: List[float] = field(default_factory=list)
    quartile_forward_returns: Dict[str, float] = field(default_factory=dict)
    n_obs: int = 0

    def as_dict(self) -> Dict:
        return {
            "deviation_pct": round(self.deviation_pct, 2),
            "fair_value": round(self.fair_value, 2),
            "spot": round(self.spot, 2),
            "half_life_days": (
                round(self.half_life_days, 1) if self.half_life_days is not None else None
            ),
            "lambda_daily": round(self.lambda_daily, 6),
            "r_squared": round(self.r_squared, 4),
            "deviation_z": round(self.deviation_z, 2),
            "band_std_pct": round(self.band_std_pct, 2),
            "regime_break": self.regime_break,
            "interpretation": self.interpretation,
            "coefficients": {k: round(v, 6) for k, v in self.coefficients.items()},
            "quartile_forward_returns": {
                k: round(v, 4) for k, v in self.quartile_forward_returns.items()
            },
            "n_obs": self.n_obs,
        }


def _interpret_deviation(deviation_pct: float, deviation_z: float) -> tuple:
    """Turn a raw deviation into an honest, caveated reading.

    Gold's real-rate/USD anchor broke structurally after 2022 (central-bank
    demand era), so a large deviation is a regime signal, NOT a clean
    over/undervaluation call. Beyond ~2 std of its own history we flag it and
    refuse to render a directional verdict.
    """
    regime_break = abs(deviation_z) >= 2.0
    if regime_break:
        side = "高于" if deviation_pct >= 0 else "低于"
        text = (
            f"当前金价{side}宏观公允值 {abs(deviation_pct):.1f}%，已超出历史正常波动带"
            f"（{abs(deviation_z):.1f} 倍标准差），处于结构性偏离期。"
            "2022 年后央行购金抬升了结构性需求，实际利率+美元锚在此机制下会系统性低估金价；"
            "该读数是机制信号，不能解读为简单的高估/低估或反转依据。"
        )
    elif abs(deviation_z) >= 1.0:
        side = "偏贵" if deviation_pct >= 0 else "偏便宜"
        text = (
            f"金价相对宏观公允值{side} {abs(deviation_pct):.1f}%"
            f"（约 {abs(deviation_z):.1f} 倍标准差），处于历史偏离带内，属正常估值波动。"
        )
    else:
        text = (
            f"金价接近宏观公允值（偏离 {deviation_pct:+.1f}%，不足 1 倍标准差），"
            "估值面无明显方向。"
        )
    return regime_break, text


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def fit_fair_value(
    raw: pd.DataFrame,
    *,
    min_obs: int = 750,
    fit_window_obs: Optional[int] = 2520,
    forward_horizon_days: int = TRADING_DAYS,
) -> Optional[FairValueResult]:
    """Fit the fair-value anchor. Returns None when macro columns are missing
    or the sample is too short -- callers must degrade explicitly.

    ``fit_window_obs`` restricts the levels regression to a trailing window
    (default ~10 years). Gold's sensitivity to real yields broke structurally
    after 2022 (central-bank demand era); an anchor fitted on 2004-2021 calls
    today's price 200% rich, which is a regime break, not a signal. A rolling
    anchor adapts while still flagging rich/cheap within the current era.
    """
    needed = {"Gold", "Real_10Y", "USD_Index"}
    if not needed.issubset(raw.columns):
        return None

    frame = raw[["Gold", "Real_10Y", "USD_Index"]].astype(float).dropna()
    if fit_window_obs:
        frame = frame.iloc[-fit_window_obs:]
    if len(frame) < min_obs:
        return None

    log_gold = np.log(frame["Gold"].values)
    real = frame["Real_10Y"].values
    log_usd = np.log(frame["USD_Index"].values)

    # Guard the linear-algebra path: on pathological synthetic inputs lstsq
    # can transiently over/underflow; results are validated downstream.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        X = np.column_stack([np.ones(len(frame)), real, log_usd])
        coef = _ols(X, log_gold)
        fitted = X @ coef
    resid = log_gold - fitted

    ss_res = float(((log_gold - fitted) ** 2).sum())
    ss_tot = float(((log_gold - log_gold.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Error-correction speed: does the deviation actually decay?
    d_log_gold = np.diff(log_gold)
    lam = float(_ols(resid[:-1].reshape(-1, 1), d_log_gold)[0])
    half_life = float(np.log(2) / -lam) if lam < 0 else None

    # Evidence card: mean forward returns by deviation quartile.
    resid_series = pd.Series(resid, index=frame.index)
    prices = frame["Gold"]
    fwd = prices.shift(-forward_horizon_days) / prices - 1.0
    quartiles = pd.qcut(resid_series, 4, labels=["q1_cheapest", "q2", "q3", "q4_richest"])
    quartile_stats = {
        str(label): float(fwd[quartiles == label].mean())
        for label in ["q1_cheapest", "q2", "q3", "q4_richest"]
        if fwd[quartiles == label].notna().any()
    }

    spot = float(frame["Gold"].iloc[-1])
    fair = float(np.exp(fitted[-1]))
    deviation_pct = (spot / fair - 1.0) * 100.0

    # Classify the deviation against its OWN historical dispersion so a large
    # number reads as "structural break", not a naive over/undervaluation call.
    resid_std = float(np.std(resid))
    band_std_pct = resid_std * 100.0
    deviation_z = float(resid[-1] / resid_std) if resid_std > 1e-9 else 0.0
    regime_break, interpretation = _interpret_deviation(deviation_pct, deviation_z)

    return FairValueResult(
        deviation_pct=deviation_pct,
        fair_value=fair,
        spot=spot,
        half_life_days=half_life,
        lambda_daily=lam,
        r_squared=r_squared,
        deviation_z=deviation_z,
        band_std_pct=band_std_pct,
        regime_break=regime_break,
        interpretation=interpretation,
        coefficients={
            "intercept": float(coef[0]),
            "real_10y": float(coef[1]),
            "log_usd": float(coef[2]),
        },
        deviation_series_tail=[float(x * 100.0) for x in resid[-260:]],
        quartile_forward_returns=quartile_stats,
        n_obs=len(frame),
    )
