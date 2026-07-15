"""Regime decision brain for GoldenSense.

Converts a gold price history + volatility state + the user's risk profile into a
stance / action / target-exposure recommendation, using the SAME signals that
were validated in the cost-after backtests:

- multi-timescale trend ensemble (1/3/12-month lookbacks; Hurst/Ooi/Pedersen)
- volatility de-risking (Moreira/Muir) via a per-vol-state exposure scale

The backtested risk menu (full-sample 2021-2026, 2 bps/side):
    conservative -> vol-managed multi-trend   Sharpe 1.09 | MaxDD  -6.8%
    balanced     -> 200d / multi trend long   Sharpe 1.21 | MaxDD -16.6%
    aggressive   -> buy & hold                 Sharpe 1.31 | MaxDD -20.4%

This is NOT an alpha engine. It is a reproducible, drawdown-aware exposure map.
The LLM narrates it; it does not invent it. Risk gates / data degradation in the
gateway still override this to 观望 when appropriate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from strategy_factors import trend_signal

# Per-profile maximum gold exposure (% of the user's intended gold allocation).
PROFILE_EXPOSURE_CAP = {"conservative": 40.0, "balanced": 70.0, "aggressive": 100.0}
# Moreira-Muir style de-risking by volatility state.
VOL_EXPOSURE_SCALE = {"calm": 1.0, "elevated": 0.7, "stress": 0.4}


@dataclass
class RegimeDecision:
    regime: str  # uptrend | downtrend | mixed
    trend_score: float  # 0..1 graded multi-trend exposure
    vol_state: str  # calm | elevated | stress
    stance: str  # 偏多 | 中性 | 偏空
    action: str  # 分批布局 | 小仓试探 | 降低暴露 | 观望
    target_exposure_pct: float  # 0..100
    confidence_band: str  # 高 | 中 | 低
    sufficient_history: bool
    reasons: List[str] = field(default_factory=list)
    # Optional probabilistic extension (evaluate_regime_v2): HMM posterior
    # {"calm": p, "elevated": p, "stress": p} and its provenance label.
    state_probabilities: Optional[dict] = None
    regime_model: str = "rule_threshold"


def _lookbacks_for(n: int):
    if n >= 252:
        return (21, 63, 252), True
    if n >= 120:
        return (21, 63, 120), True
    if n >= 60:
        return (10, 20, 60), True
    if n >= 10:
        return (3, 5, max(6, n // 2)), False
    return (2, 3, 4), False


def _multi_trend_score(prices: pd.Series, lookbacks) -> float:
    subs = [trend_signal(prices, lb).iloc[-1] for lb in lookbacks]
    return float(np.mean(subs))


def _infer_vol_state(prices: pd.Series) -> str:
    rets = prices.pct_change().dropna()
    if len(rets) < 20:
        return "elevated"
    ann_vol = float(rets.tail(20).std() * np.sqrt(252))
    if ann_vol >= 0.22:
        return "stress"
    if ann_vol >= 0.14:
        return "elevated"
    return "calm"


def evaluate_regime(
    prices: pd.Series,
    *,
    risk_profile: str,
    vol_state: Optional[str] = None,
) -> RegimeDecision:
    profile = risk_profile if risk_profile in PROFILE_EXPOSURE_CAP else "balanced"
    if not isinstance(prices, pd.Series):
        prices = pd.Series(list(prices), dtype=float)
    prices = prices.astype(float).dropna()
    n = len(prices)

    lookbacks, sufficient = _lookbacks_for(n)
    trend_score = _multi_trend_score(prices, lookbacks) if n >= 2 else 0.5

    if vol_state not in VOL_EXPOSURE_SCALE:
        vol_state = _infer_vol_state(prices)

    if trend_score >= 0.66:
        regime, stance = "uptrend", "偏多"
        action = "小仓试探" if profile == "conservative" else "分批布局"
    elif trend_score <= 0.34:
        regime, stance = "downtrend", "偏空"
        action = "降低暴露"
    else:
        regime, stance = "mixed", "中性"
        action = "观望"

    vol_scale = VOL_EXPOSURE_SCALE[vol_state]
    cap = PROFILE_EXPOSURE_CAP[profile]
    target_exposure_pct = round(cap * trend_score * vol_scale, 1)
    if regime == "downtrend":
        target_exposure_pct = 0.0
    target_exposure_pct = float(min(100.0, max(0.0, target_exposure_pct)))

    full_agreement = trend_score in (0.0, 1.0)
    if not sufficient:
        confidence_band = "低"
    elif vol_state == "stress":
        confidence_band = "低"
    elif full_agreement and vol_state == "calm":
        confidence_band = "高"
    else:
        confidence_band = "中"

    horizon_lbl = "/".join(str(lb) for lb in lookbacks)
    reasons = [
        f"多周期趋势({horizon_lbl}日)综合得分 {trend_score:.2f}，判定为{regime}。",
        f"波动状态 {vol_state}，按 Moreira-Muir 波动率管理对暴露打 {vol_scale:.0%} 折扣。",
        f"{profile} 画像的研究参考黄金暴露上限 {cap:.0f}%，本轮参考暴露约 {target_exposure_pct:.0f}%（非投资建议）。",
    ]
    if not sufficient:
        reasons.append("历史样本不足以稳定估计长周期趋势，已按低置信度处理。")

    return RegimeDecision(
        regime=regime,
        trend_score=round(trend_score, 4),
        vol_state=vol_state,
        stance=stance,
        action=action,
        target_exposure_pct=target_exposure_pct,
        confidence_band=confidence_band,
        sufficient_history=sufficient,
        reasons=reasons,
    )


def evaluate_regime_v2(
    prices: pd.Series,
    *,
    risk_profile: str,
    vol_state: Optional[str] = None,
    state_posterior: Optional[dict] = None,
) -> RegimeDecision:
    """Probability-blended regime decision.

    When an HMM ``state_posterior`` ({"calm": p, "elevated": p, "stress": p},
    typically fitted on the long local history by ``regime_probabilistic``) is
    supplied, the volatility de-risking scale becomes the probability-weighted
    blend of the per-state scales instead of a hard bucket, which removes
    flip-flopping near regime borders. Without a posterior this falls back to
    the rule-threshold ``evaluate_regime`` unchanged.
    """
    if not state_posterior:
        return evaluate_regime(prices, risk_profile=risk_profile, vol_state=vol_state)

    # Normalize the posterior defensively (it travels through JSON).
    probs = {k: max(0.0, float(v)) for k, v in state_posterior.items() if k in VOL_EXPOSURE_SCALE}
    total = sum(probs.values())
    if total <= 0:
        return evaluate_regime(prices, risk_profile=risk_profile, vol_state=vol_state)
    probs = {k: v / total for k, v in probs.items()}

    base = evaluate_regime(prices, risk_profile=risk_profile, vol_state=vol_state)

    blended_scale = sum(VOL_EXPOSURE_SCALE[k] * p for k, p in probs.items())
    dominant_state = max(probs, key=probs.get)
    dominant_prob = probs[dominant_state]

    profile = risk_profile if risk_profile in PROFILE_EXPOSURE_CAP else "balanced"
    cap = PROFILE_EXPOSURE_CAP[profile]
    target = round(cap * base.trend_score * blended_scale, 1)
    if base.regime == "downtrend":
        target = 0.0
    target = float(min(100.0, max(0.0, target)))

    # Confidence: posterior concentration replaces the hard vol_state rule.
    if not base.sufficient_history or (dominant_state == "stress" and dominant_prob >= 0.5):
        confidence_band = "低"
    elif dominant_prob >= 0.75 and base.trend_score in (0.0, 1.0):
        confidence_band = "高"
    else:
        confidence_band = "中"

    prob_txt = "、".join(f"{k} {v:.0%}" for k, v in sorted(probs.items(), key=lambda x: -x[1]))
    reasons = list(base.reasons[:1])
    reasons.append(
        f"HMM 状态后验：{prob_txt}；按概率加权后的波动折扣为 {blended_scale:.0%}。"
    )
    reasons.append(
        f"{profile} 画像暴露上限 {cap:.0f}%，概率混合后目标暴露约 {target:.0f}%。"
    )
    if not base.sufficient_history:
        reasons.append("历史样本不足以稳定估计长周期趋势，已按低置信度处理。")

    return RegimeDecision(
        regime=base.regime,
        trend_score=base.trend_score,
        vol_state=dominant_state,
        stance=base.stance,
        action=base.action,
        target_exposure_pct=target,
        confidence_band=confidence_band,
        sufficient_history=base.sufficient_history,
        reasons=reasons,
        state_probabilities={k: round(v, 4) for k, v in probs.items()},
        regime_model="hmm_posterior_blend",
    )
