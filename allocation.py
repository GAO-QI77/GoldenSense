"""Strategic allocation layer: Black-Litterman-lite tilts + regime-switching
Monte Carlo scenario cones.

Design
------
- The user's risk-profile questionnaire defines the *prior* portfolio weight
  band for gold (the market-equilibrium stand-in). Model signals never invent
  an allocation from scratch; they only tilt the prior, with hard caps -- the
  spirit of Black-Litterman without pretending we have a full covariance view.
- Views come from two validated sources: the HMM regime posterior (risk
  appetite) and the fair-value deviation (valuation). Both tilt multiplied,
  both bounded, fully deterministic and auditable.
- The T+30/T+90 outlook is a *distribution*, not a point estimate: a
  regime-switching Monte Carlo cone sampled from the fitted HMM (per-state
  return mean/vol + transition matrix), so scenario width honestly reflects
  the current regime mix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from regime_probabilistic import GaussianHMM, STATE_LABELS, regime_features

# Prior gold weight band (% of total portfolio) per risk profile. Conventional
# strategic ranges used by allocators; the questionnaire prior, not a signal.
PROFILE_PRIOR_RANGE = {
    "conservative": (2.0, 8.0),
    "balanced": (5.0, 12.0),
    "aggressive": (8.0, 18.0),
}

# View strength caps: tilts can move the band by at most this fraction.
MAX_REGIME_TILT = 0.25
MAX_VALUATION_TILT = 0.25


# Fixed de-advice disclaimer attached to every allocation output.
ALLOCATION_DISCLAIMER = (
    "以下为教育型研究参考区间，基于公开宏观关系与历史统计，"
    "不构成任何投资建议、要约或个性化理财意见；实际决策请咨询持牌顾问并自担风险。"
)


@dataclass
class AllocationAdvice:
    profile: str
    prior_range_pct: Sequence[float]
    reference_range_pct: Sequence[float]
    regime_tilt: float
    valuation_tilt: float
    rationale: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "profile": self.profile,
            "prior_range_pct": [round(x, 1) for x in self.prior_range_pct],
            # Kept under the legacy key for API compatibility, but framed as a
            # research reference range (not a recommendation) everywhere in copy.
            "reference_range_pct": [round(x, 1) for x in self.reference_range_pct],
            "recommended_range_pct": [round(x, 1) for x in self.reference_range_pct],
            "regime_tilt": round(self.regime_tilt, 4),
            "valuation_tilt": round(self.valuation_tilt, 4),
            "rationale": list(self.rationale),
            "disclaimer": ALLOCATION_DISCLAIMER,
        }


def regime_tilt_from_posterior(posterior: Optional[Dict[str, float]]) -> float:
    """Map the HMM posterior to a tilt in [-MAX_REGIME_TILT, +MAX_REGIME_TILT].

    Calm regimes carry a small positive tilt (risk budget available), stress
    regimes a negative one (de-risking), elevated neutral.
    """
    if not posterior:
        return 0.0
    weights = {"calm": 1.0, "elevated": 0.0, "stress": -1.0}
    total = sum(max(0.0, float(p)) for p in posterior.values()) or 1.0
    tilt = sum(
        weights.get(state, 0.0) * max(0.0, float(p)) / total
        for state, p in posterior.items()
    )
    return float(np.clip(tilt * MAX_REGIME_TILT, -MAX_REGIME_TILT, MAX_REGIME_TILT))


def valuation_tilt_from_deviation(deviation_pct: Optional[float]) -> float:
    """Fair-value deviation -> tilt. ±30% deviation saturates the ±25% tilt.

    Rich vs macro fair value tilts the band down; cheap tilts it up.
    """
    if deviation_pct is None:
        return 0.0
    return float(np.clip(-deviation_pct / 30.0, -1.0, 1.0) * MAX_VALUATION_TILT)


def allocation_range(
    risk_profile: str,
    *,
    regime_posterior: Optional[Dict[str, float]] = None,
    valuation_deviation_pct: Optional[float] = None,
) -> AllocationAdvice:
    profile = risk_profile if risk_profile in PROFILE_PRIOR_RANGE else "balanced"
    lo, hi = PROFILE_PRIOR_RANGE[profile]

    r_tilt = regime_tilt_from_posterior(regime_posterior)
    v_tilt = valuation_tilt_from_deviation(valuation_deviation_pct)
    combined = float(np.clip(1.0 + r_tilt + v_tilt, 0.5, 1.5))

    rec_lo = max(0.0, lo * combined)
    rec_hi = min(25.0, hi * combined)

    rationale = [
        f"{profile} 画像的战略先验区间为组合的 {lo:.0f}%–{hi:.0f}%（问卷先验，非信号）。"
    ]
    if regime_posterior:
        rationale.append(f"HMM 状态观点带来 {r_tilt:+.0%} 倾斜（calm 加、stress 减，上限 ±25%）。")
    if valuation_deviation_pct is not None:
        direction = "高于" if valuation_deviation_pct >= 0 else "低于"
        rationale.append(
            f"金价{direction}宏观公允值 {abs(valuation_deviation_pct):.1f}%，"
            f"估值观点带来 {v_tilt:+.0%} 倾斜（±30% 偏离饱和）。"
        )
    rationale.append(
        f"综合研究参考区间 {rec_lo:.1f}%–{rec_hi:.1f}%；观点只倾斜先验、不替代先验（BL 纪律）。"
        "该区间为研究口径，非投资建议。"
    )

    return AllocationAdvice(
        profile=profile,
        prior_range_pct=(lo, hi),
        reference_range_pct=(rec_lo, rec_hi),
        regime_tilt=r_tilt,
        valuation_tilt=v_tilt,
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# Regime-switching Monte Carlo cone
# --------------------------------------------------------------------------- #
def monte_carlo_cone(
    prices: pd.Series,
    *,
    horizon_days: int = 90,
    n_paths: int = 1500,
    n_states: int = 3,
    seed: int = 7,
    model: Optional[GaussianHMM] = None,
    checkpoints: Sequence[int] = (30, 90),
) -> Optional[Dict]:
    """Simulate price paths from the fitted HMM; return percentile cone.

    Returns None when history is too short to fit the regime model. The cone
    is a distributional statement ("30 天后 80% 概率落在 X–Y"), which is the
    honest replacement for a T+30 point forecast.
    """
    feats = regime_features(prices)
    if len(feats) < 400:
        return None

    if model is None or model.fit_ is None:
        model = GaussianHMM(n_states=n_states, max_iter=40).fit(feats.values)
    fit = model.fit_

    posterior = model.posterior(feats.values)[-1]
    ret_means = fit.means[:, 0]
    ret_stds = np.sqrt(fit.variances[:, 0])
    transition = fit.transition

    rng = np.random.default_rng(seed)
    spot = float(prices.dropna().iloc[-1])
    n_k = len(ret_means)

    states = rng.choice(n_k, size=n_paths, p=posterior / posterior.sum())
    log_prices = np.full(n_paths, np.log(spot))
    path_percentiles: Dict[str, List[float]] = {"p10": [], "p50": [], "p90": []}
    checkpoint_stats: Dict[str, Dict[str, float]] = {}

    for day in range(1, horizon_days + 1):
        # Advance the hidden state, then draw a return conditional on it.
        u = rng.random(n_paths)
        cum = transition[states].cumsum(axis=1)
        states = (u[:, None] > cum).sum(axis=1).clip(0, n_k - 1)
        draws = rng.normal(ret_means[states], ret_stds[states])
        log_prices = log_prices + np.log1p(np.clip(draws, -0.5, 0.5))

        level = np.exp(log_prices)
        path_percentiles["p10"].append(float(np.percentile(level, 10)))
        path_percentiles["p50"].append(float(np.percentile(level, 50)))
        path_percentiles["p90"].append(float(np.percentile(level, 90)))

        if day in checkpoints:
            checkpoint_stats[f"d{day}"] = {
                "p10": float(np.percentile(level, 10)),
                "p50": float(np.percentile(level, 50)),
                "p90": float(np.percentile(level, 90)),
                "prob_above_spot": float((level > spot).mean()),
            }

    labels = [STATE_LABELS.get(k, f"state_{k}") for k in range(n_k)]
    return {
        "spot": spot,
        "horizon_days": horizon_days,
        "n_paths": n_paths,
        "start_posterior": {labels[k]: round(float(posterior[k]), 4) for k in range(n_k)},
        "percentile_paths": path_percentiles,
        "checkpoints": checkpoint_stats,
    }
