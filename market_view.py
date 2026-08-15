"""Unified market view book: one structured document, three time horizons.

The quantitative research context already computes every validated block
(vol bands, HMM posterior, macro factors, fair value, scenario cone,
flagship). This module *assembles* them into the product's headline output --
a short/mid/long "view book" -- without recomputing or inventing a single
number. Each horizon section states a core view, a confidence grade, the
evidence behind it, and -- deliberately first-class -- the *invalidation
conditions*: what observable change would void the view. A view that cannot
say how it dies is marketing, not research.

Short-term is expressed as distribution + risk only: directional short-term
prediction was falsified in walk-forward validation and is intentionally
absent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# 21-day return band width (p90 - p10) above this -> low short-term confidence.
SHORT_TERM_WIDE_BAND = 0.08
# Dominant regime posterior thresholds for mid-term confidence grading.
MID_CONF_HIGH = 0.75
MID_CONF_MED = 0.60
# |z| at/above this marks a structural valuation deviation (matches fair_value).
LONG_TERM_STRUCTURAL_Z = 2.0
LONG_TERM_STRUCTURAL_PCT = 50.0

_STATE_ZH = {"calm": "平静", "elevated": "抬升", "stress": "压力"}


def _unavailable(reason: str) -> Dict[str, Any]:
    return {"available": False, "degraded_reason": reason}


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


# --------------------------------------------------------------------------- #
def _short_term(ctx: Dict[str, Any]) -> Dict[str, Any]:
    bands = ctx.get("vol_bands")
    if not bands or "h21" not in bands:
        return _unavailable(str(ctx.get("degraded", {}).get("vol_bands", "vol_bands_missing")))

    h21 = bands["h21"]
    width = float(h21["p90"]) - float(h21["p10"])
    confidence = "低" if width > SHORT_TERM_WIDE_BAND else "中"

    evidence: List[str] = []
    for key, label in (("h1", "1 天"), ("h5", "5 天"), ("h21", "21 天")):
        b = bands.get(key)
        if b:
            evidence.append(
                f"{label}收益分布带 {_pct(float(b['p10']))} ~ {_pct(float(b['p90']))}"
                f"（中位 {_pct(float(b['p50']))}）。"
            )
    ann_vol = h21.get("ann_vol_forecast")
    if ann_vol is not None:
        evidence.append(f"HAR-RV 年化波动率预测 {float(ann_vol) * 100:.1f}%。")

    core_view = (
        f"短期（1-21 天）以分布与风险为口径：未来 21 天收益 80% 区间为 "
        f"{_pct(float(h21['p10']))} ~ {_pct(float(h21['p90']))}。"
        "本系统不发布短期方向观点——方向预测在走前验证中无显著边际，如实不提供。"
    )
    if width > SHORT_TERM_WIDE_BAND:
        core_view += "当前分布带偏宽，短期不确定性高于常态。"

    return {
        "available": True,
        "core_view": core_view,
        "confidence": confidence,
        "evidence": evidence,
        "invalidation": [
            "已实现波动显著突破 HAR-RV 预测带（覆盖率诊断连续失效）时，区间口径作废。",
            f"21 天分布带宽突破 {SHORT_TERM_WIDE_BAND:.0%} 阈值时，置信度降档。",
        ],
        "data": {"bands": bands},
    }


def _mid_term(ctx: Dict[str, Any]) -> Dict[str, Any]:
    regime = ctx.get("regime_posterior")
    factors = ctx.get("macro_factors")
    if not regime or not regime.get("latest"):
        return _unavailable(str(ctx.get("degraded", {}).get("regime_posterior", "regime_missing")))
    if not factors:
        return _unavailable(str(ctx.get("degraded", {}).get("macro_factors", "macro_factors_missing")))

    latest: Dict[str, float] = regime["latest"]
    dominant = max(latest, key=lambda k: float(latest[k]))
    p_dom = float(latest[dominant])
    if p_dom >= MID_CONF_HIGH:
        confidence = "高"
    elif p_dom >= MID_CONF_MED:
        confidence = "中"
    else:
        confidence = "低"

    composite = float(factors.get("composite", 0.5))
    if composite >= 0.55:
        factor_view = f"宏观因子组合得分 {composite:.2f}，构成顺风。"
    elif composite <= 0.45:
        factor_view = f"宏观因子组合得分 {composite:.2f}，构成逆风。"
    else:
        factor_view = f"宏观因子组合得分 {composite:.2f}，方向中性。"

    core_view = (
        f"中期（1-6 月）主导状态为「{_STATE_ZH.get(dominant, dominant)}」"
        f"（概率 {p_dom:.0%}）。{factor_view}"
    )

    evidence = [
        "HMM 概率状态：" + "、".join(
            f"{_STATE_ZH.get(k, k)} {float(v):.0%}" for k, v in latest.items()
        ) + "。",
        f"因子口径：{'、'.join(factors.get('factors_used', []))}。",
    ]

    return {
        "available": True,
        "core_view": core_view,
        "confidence": confidence,
        "evidence": evidence,
        "invalidation": [
            f"主导状态概率跌破 {MID_CONF_MED:.0%} 时，中期观点降级为低置信。",
            "因子复合得分翻越 0.5 中轴时，顺/逆风判断作废。",
        ],
        "data": {"regime_latest": latest, "macro_composite": composite},
    }


def _long_term(ctx: Dict[str, Any]) -> Dict[str, Any]:
    fair = ctx.get("fair_value")
    if not fair:
        return _unavailable(str(ctx.get("degraded", {}).get("fair_value", "fair_value_missing")))

    deviation_pct = fair.get("deviation_pct")
    deviation_z = fair.get("deviation_z")
    structural = bool(fair.get("regime_break")) or (
        deviation_z is not None and abs(float(deviation_z)) >= LONG_TERM_STRUCTURAL_Z
    ) or (
        deviation_pct is not None and abs(float(deviation_pct)) >= LONG_TERM_STRUCTURAL_PCT
    )

    if deviation_pct is None:
        return _unavailable("fair_value_deviation_missing")

    direction = "高于" if float(deviation_pct) >= 0 else "低于"
    if structural:
        valuation_view = (
            f"金价{direction}宏观公允锚 {abs(float(deviation_pct)):.1f}%"
            f"（{float(deviation_z):+.1f}σ），处于结构性偏离期——历史锚定关系可能正在改写。"
        )
        confidence = "低"
    else:
        z_txt = f"（{float(deviation_z):+.1f}σ）" if deviation_z is not None else ""
        valuation_view = (
            f"金价{direction}宏观公允锚 {abs(float(deviation_pct)):.1f}%{z_txt}，"
            "仍在正常估值波动带内。"
        )
        confidence = "中"

    evidence = [valuation_view]
    cone = ctx.get("scenario_cone") or {}
    d90 = (cone.get("checkpoints") or {}).get("d90")
    if d90:
        evidence.append(
            f"90 天蒙特卡洛情景锥：p10 {d90['p10']:.0f} / p50 {d90['p50']:.0f} / "
            f"p90 {d90['p90']:.0f}，高于现价概率 {float(d90.get('prob_above_spot', 0)):.0%}。"
        )
    flagship = ctx.get("flagship") or {}
    metrics = flagship.get("metrics") or {}
    if metrics:
        evidence.append(
            f"旗舰策略（趋势×实际利率×HMM 去险×波动目标）全样本 Sharpe "
            f"{metrics.get('sharpe')}，最大回撤 {float(metrics.get('max_drawdown', 0)) * 100:.0f}%。"
        )

    core_view = f"长期（6 月+）估值口径：{valuation_view}"

    return {
        "available": True,
        "core_view": core_view,
        "confidence": confidence,
        "evidence": evidence,
        "invalidation": [
            f"公允价值偏离 z 分突破 ±{LONG_TERM_STRUCTURAL_Z:.0f} 时，正常带判断作废（进入结构性偏离口径）。",
            "宏观锚回归关系失效（regime_break 标记）时，长期估值观点整体作废。",
        ],
        "data": {
            "deviation_pct": deviation_pct,
            "deviation_z": deviation_z,
            "regime_break": fair.get("regime_break"),
        },
    }


# --------------------------------------------------------------------------- #
def build_market_view(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the three-horizon view book from a research-context dict.

    Pure assembly: every number in the output already exists in ``ctx``.
    Sections degrade independently and explicitly.
    """
    return {
        "short_term": _short_term(ctx),
        "mid_term": _mid_term(ctx),
        "long_term": _long_term(ctx),
        "meta": {
            "data_source": ctx.get("data_source"),
            "data_asof": ctx.get("data_asof"),
            "data_age_days": ctx.get("data_age_days"),
            "data_stale": ctx.get("data_stale"),
            "is_realtime": ctx.get("is_realtime", False),
            "degraded": dict(ctx.get("degraded", {})),
        },
    }


def summarize_for_ledger(book: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """One-line-per-horizon summary frozen into weekly signal publications."""
    out: Dict[str, Optional[str]] = {}
    for horizon in ("short_term", "mid_term", "long_term"):
        section = book.get(horizon) or {}
        out[horizon] = section.get("core_view") if section.get("available") else None
    return out
