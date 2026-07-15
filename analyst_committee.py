"""Deterministic analyst committee: four specialist views, one auditable fusion.

Instead of a single opaque score, the gateway now assembles four *specialist
analysts* -- technical, macro, flow, news/event. Each is a deterministic
function that reads only its own slice of evidence and emits:

    stance      in [-1, +1]   (bearish .. bullish)
    confidence  in [0, 1]
    evidence    list[str]     (human-readable, cites its inputs)

The aggregator fuses stances with regime-dependent weights and reports a
**disagreement score** (confidence-weighted dispersion). Disagreement is a
first-class signal: it routes the narrator to the stronger model, lowers the
confidence band, and is stored in the trace. No LLM is involved anywhere in
this module -- narration happens downstream and can only cite these numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Regime-conditional analyst weights: in stress, macro/flow dominate price
# trend; in calm regimes technicals carry more.
REGIME_WEIGHTS = {
    "calm": {"technical": 0.35, "macro": 0.30, "flow": 0.15, "news": 0.20},
    "elevated": {"technical": 0.30, "macro": 0.35, "flow": 0.15, "news": 0.20},
    "stress": {"technical": 0.20, "macro": 0.40, "flow": 0.15, "news": 0.25},
}
DISAGREEMENT_CONFLICT_THRESHOLD = 0.35


@dataclass
class AnalystView:
    name: str
    stance: float
    confidence: float
    evidence: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stance": round(float(np.clip(self.stance, -1.0, 1.0)), 4),
            "confidence": round(float(np.clip(self.confidence, 0.0, 1.0)), 4),
            "evidence": list(self.evidence),
        }


def technical_analyst(regime: Optional[Dict[str, Any]]) -> AnalystView:
    """Reads the validated trend/vol regime decision."""
    if not regime:
        return AnalystView("technical", 0.0, 0.1, ["无可用趋势状态数据。"])
    trend_score = float(regime.get("trend_score", 0.5))
    stance = (trend_score - 0.5) * 2.0
    confidence = 0.7 if regime.get("sufficient_history") else 0.3
    evidence = [
        f"多周期趋势得分 {trend_score:.2f}，趋势状态 {regime.get('regime')}。",
        f"波动状态 {regime.get('vol_state')}，目标暴露 {regime.get('target_exposure_pct')}%。",
    ]
    return AnalystView("technical", stance, confidence, evidence)


def macro_analyst(
    macro_factors: Optional[Dict[str, Any]],
    fair_value: Optional[Dict[str, Any]],
) -> AnalystView:
    """Reads the factor sleeve (real rates / USD / inflation) + valuation."""
    if not macro_factors:
        return AnalystView("macro", 0.0, 0.1, ["宏观因子数据不可用。"])
    composite = float(macro_factors.get("composite", 0.5))
    stance = (composite - 0.5) * 2.0
    confidence = min(0.75, 0.25 + 0.125 * len(macro_factors.get("factors_used", [])))
    evidence = [
        f"宏观因子组合得分 {composite:.2f}"
        f"（{'、'.join(macro_factors.get('factors_used', []))}）。"
    ]
    if fair_value and fair_value.get("deviation_pct") is not None:
        deviation = float(fair_value["deviation_pct"])
        # Valuation is a slow anchor: it dampens rather than flips the stance.
        stance -= float(np.clip(deviation / 30.0, -1.0, 1.0)) * 0.3
        direction = "高于" if deviation >= 0 else "低于"
        evidence.append(
            f"金价{direction}宏观公允值 {abs(deviation):.1f}%（实际利率+美元锚定）。"
        )
    return AnalystView("macro", stance, confidence, evidence)


def flow_analyst(macro_factors: Optional[Dict[str, Any]]) -> AnalystView:
    """Reads the flow-confirmation proxy factor."""
    latest = (macro_factors or {}).get("latest", {})
    if "flow_confirmation" not in latest:
        return AnalystView("flow", 0.0, 0.1, ["资金流代理数据不可用。"])
    flow = float(latest["flow_confirmation"])
    stance = (flow - 0.5) * 1.2  # proxy data -> deliberately muted stance
    evidence = [
        "GLD 美元成交量确认上升趋势。" if flow >= 0.5 else "资金流代理未确认当前趋势。",
        "注：该因子为成交额代理，非真实 ETF 持仓流。",
    ]
    return AnalystView("flow", stance, 0.4, evidence)


def news_analyst(
    news_sentiment: Optional[float],
    vix_value: Optional[float],
    vix_threshold: float = 30.0,
) -> AnalystView:
    """Reads scored news sentiment plus the VIX risk backdrop."""
    if news_sentiment is None:
        return AnalystView("news", 0.0, 0.1, ["新闻情绪数据不可用。"])
    stance = float(np.clip(news_sentiment, -1.0, 1.0))
    confidence = 0.5
    evidence = [f"近端新闻情绪得分 {news_sentiment:+.2f}。"]
    if vix_value is not None:
        if vix_value >= vix_threshold:
            stance = min(stance, 0.0)
            confidence = 0.35
            evidence.append(f"VIX {vix_value:.1f} 高于熔断阈值 {vix_threshold:.0f}，风险压制多头观点。")
        else:
            evidence.append(f"VIX {vix_value:.1f} 处于阈值之下。")
    return AnalystView("news", stance, confidence, evidence)


# --------------------------------------------------------------------------- #
def build_committee(
    *,
    regime: Optional[Dict[str, Any]],
    macro_factors: Optional[Dict[str, Any]] = None,
    fair_value: Optional[Dict[str, Any]] = None,
    news_sentiment: Optional[float] = None,
    vix_value: Optional[float] = None,
    vix_threshold: float = 30.0,
) -> Dict[str, Any]:
    """Assemble views, fuse them with regime weights, score disagreement."""
    views = [
        technical_analyst(regime),
        macro_analyst(macro_factors, fair_value),
        flow_analyst(macro_factors),
        news_analyst(news_sentiment, vix_value, vix_threshold),
    ]

    vol_state = (regime or {}).get("vol_state", "elevated")
    weights_map = REGIME_WEIGHTS.get(vol_state, REGIME_WEIGHTS["elevated"])

    # Effective weight = regime weight x analyst confidence, renormalized.
    raw_weights = np.array([weights_map[v.name] * max(v.confidence, 1e-6) for v in views])
    weights = raw_weights / raw_weights.sum()
    stances = np.array([np.clip(v.stance, -1.0, 1.0) for v in views])

    fused = float(np.dot(weights, stances))
    disagreement = float(np.sqrt(np.dot(weights, (stances - fused) ** 2)))

    if fused >= 0.15:
        fused_label = "偏多"
    elif fused <= -0.15:
        fused_label = "偏空"
    else:
        fused_label = "中性"

    return {
        "views": [v.as_dict() for v in views],
        "weights": {v.name: round(float(w), 4) for v, w in zip(views, weights)},
        "regime_weight_profile": vol_state,
        "fused_stance": round(fused, 4),
        "fused_label": fused_label,
        "disagreement": round(disagreement, 4),
        "has_material_disagreement": disagreement >= DISAGREEMENT_CONFLICT_THRESHOLD,
    }
