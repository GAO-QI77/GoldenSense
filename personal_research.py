"""Personalized research rule engine: deterministic numbers, bounded framing.

Second pillar of the product. Given an investor profile and the cached
research context, this module produces every *number* of the personalized
output -- reference range, position gap, structured risk flags, horizon-
matched evidence -- as pure rules over already-validated blocks. The LLM
downstream only rewrites language around these facts and is double-gated:
the narrative critic grounds its numbers, and the directive-language check
keeps the output inside the "reference range / gap / risk note" framing.
Nothing here (or downstream) tells anyone to buy or sell anything.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from allocation import ALLOCATION_DISCLAIMER
from investor_profile import InvestorProfile, PersonalNarrative
from market_view import build_market_view

# Rule thresholds (auditable, deliberately few).
STRESS_PROB_FLAG = 0.35
FAR_ABOVE_RANGE_PCT = 5.0
SHORT_HORIZON_BAND_WIDTH = 0.08
STRUCTURAL_Z = 2.0

_HORIZON_TO_SECTION = {"short": "short_term", "mid": "mid_term", "long": "long_term"}

_RISK_TOLERANCE_ZH = {"conservative": "保守", "balanced": "稳健", "aggressive": "进取"}
_HORIZON_ZH = {"short": "短期", "mid": "中期", "long": "长期"}

# Directive / imperative trading language is out of product scope. Patterns are
# deliberately broad: a false positive falls back to the compliant draft, which
# is the safe failure mode.
_DIRECTIVE_PATTERNS = [
    r"你应该\s*(买|卖|加仓|减仓|持有)",
    r"您应该\s*(买|卖|加仓|减仓|持有)",
    r"建议\s*(你|您)?\s*(买入|卖出|加仓|减仓|清仓|建仓)",
    r"立即\s*(买|卖|加仓|减仓|清仓|建仓)",
    r"(赶紧|马上|尽快)\s*(买|卖|加仓|减仓|清仓)",
    r"满仓",
    r"梭哈",
    r"必须\s*(买|卖|持有)",
    r"抄底",
    r"(买入|卖出)信号",
]
_DIRECTIVE_RES = [re.compile(p) for p in _DIRECTIVE_PATTERNS]


def check_no_directive_language(texts: List[str]) -> Tuple[bool, List[str]]:
    """Return (passed, violations). Any directive phrasing fails the gate."""
    violations: List[str] = []
    for text in texts:
        for pattern in _DIRECTIVE_RES:
            match = pattern.search(text or "")
            if match:
                violations.append(f"{match.group(0)!r} in {text[:60]!r}")
    return (not violations), violations


def build_three_dimensional_brief(
    profile: InvestorProfile,
    ctx: Dict[str, Any],
    research_case: Any,
) -> Dict[str, Any]:
    """Join Agent interpretation, hard rules and live API/model state.

    The returned object intentionally excludes the investor profile so it can
    be attached to a case without persisting personal inputs server-side.
    """
    facts = build_personal_facts(profile, ctx)
    horizon_key = _HORIZON_TO_SECTION[profile.horizon]
    strategy = research_case.horizon_strategy.get(horizon_key)
    if strategy is None:
        core = "当前期限策略降级，本次只保留风险观察清单。"
        scenarios: List[Dict[str, Any]] = []
        triggers: List[str] = []
        invalidation: List[str] = []
        next_review_at = research_case.created_at.isoformat()
    else:
        core = strategy.base.description
        scenarios = [
            strategy.base.model_dump(), strategy.upside.model_dump(), strategy.downside.model_dump()
        ]
        triggers = list(strategy.triggers)
        invalidation = list(strategy.invalidation)
        next_review_at = strategy.next_review_at.isoformat()

    hard_constraints = [
        "输出仅是研究行动清单，不构成直接买卖或目标仓位指令。",
        "任何模型都不能覆盖证据门控、数据陈旧标记或个人风险约束。",
    ]
    if profile.leverage_attitude in {"medium", "high"}:
        hard_constraints.append("本系统不对杠杆黄金暴露给出配置结论。")

    model_states = [
        {
            "model_id": card.model_id,
            "label": card.label,
            "state": card.state,
            "role": card.role,
            "can_influence_strategy": card.can_influence_strategy,
            "degradation_reason": card.degradation_reason,
        }
        for card in research_case.model_registry
    ]
    brief = {
        "case_id": research_case.case_id,
        "agent": {
            "core_conclusion": core,
            "scenario_focus": scenarios,
            "experience_mode": profile.experience,
        },
        "rules": {
            "risk_flags": facts.get("risk_flags", []),
            "hard_constraints": hard_constraints,
            "reference_range": facts.get("reference_range"),
            "position_gap": facts.get("position_gap"),
        },
        "api": {
            "data_asof": ctx.get("data_asof"),
            "data_age_days": ctx.get("data_age_days"),
            "freshness": "stale" if ctx.get("data_stale") else "current",
            "is_realtime": bool(ctx.get("is_realtime", False)),
            "model_states": model_states,
            "evidence_status": research_case.status,
        },
        "watchlist": triggers,
        "invalidation": invalidation,
        "next_review_at": next_review_at,
        "degradation_flags": sorted(set(
            list((facts.get("degraded") or {}).values())
            + list(research_case.audit_report.issues)
        )),
        "disclaimer": facts.get("disclaimer", ALLOCATION_DISCLAIMER),
    }
    texts = [
        str(brief["agent"]["core_conclusion"]),
        *brief["watchlist"], *brief["invalidation"], *hard_constraints,
    ]
    safe, violations = check_no_directive_language(texts)
    if not safe:  # deterministic output should never reach this branch
        raise ValueError("three-dimensional brief violated directive gate: " + "; ".join(violations))
    return brief


# --------------------------------------------------------------------------- #
def build_personal_facts(profile: InvestorProfile, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """All personalized numbers, from validated blocks only."""
    degraded: Dict[str, str] = {}
    book = build_market_view(ctx)

    # Reference range for this risk profile (allocation.py already computed it).
    adv = (ctx.get("allocation") or {}).get(profile.risk_tolerance) or {}
    rng = adv.get("reference_range_pct") or adv.get("recommended_range_pct")
    if rng and len(rng) == 2:
        lo, hi = float(rng[0]), float(rng[1])
        reference_range: Dict[str, Any] = {
            "available": True,
            "range_pct": [lo, hi],
            "midpoint": round((lo + hi) / 2.0, 2),
            "evidence_ref": f"allocation.{profile.risk_tolerance}.reference_range_pct",
        }
    else:
        reference_range = {"available": False, "evidence_ref": None}
        degraded["reference_range"] = "allocation_block_missing"
        lo = hi = None  # type: ignore[assignment]

    # Position gap vs the range.
    position_gap: Dict[str, Any]
    if lo is not None:
        current = float(profile.current_gold_pct)
        if current < lo:
            position_gap = {"status": "below", "gap_pct": round(lo - current, 2)}
        elif current > hi:
            position_gap = {"status": "above", "gap_pct": round(current - hi, 2)}
        else:
            position_gap = {"status": "within", "gap_pct": 0.0}
        position_gap["current_gold_pct"] = current
        position_gap["evidence_ref"] = "profile.current_gold_pct vs reference_range"
    else:
        position_gap = {"status": "unknown", "gap_pct": None,
                        "current_gold_pct": float(profile.current_gold_pct),
                        "evidence_ref": None}

    # Structured risk flags.
    risk_flags: List[Dict[str, Any]] = []
    stress_p = float(
        ((ctx.get("regime_posterior") or {}).get("latest") or {}).get("stress", 0.0)
    )
    if position_gap["status"] == "above" and stress_p >= STRESS_PROB_FLAG:
        risk_flags.append({
            "flag": "position_above_range_in_stress",
            "detail": (
                f"当前仓位 {profile.current_gold_pct:.0f}% 高于参考区间上沿，"
                f"且市场压力状态概率 {stress_p:.0%} 处于高位。"
            ),
            "evidence_ref": "regime_posterior.latest.stress",
        })
    if position_gap["status"] == "above" and (position_gap["gap_pct"] or 0) > FAR_ABOVE_RANGE_PCT:
        risk_flags.append({
            "flag": "position_far_above_range",
            "detail": (
                f"当前仓位超出参考区间上沿 {position_gap['gap_pct']:.1f} 个百分点，"
                f"集中度显著高于 {_RISK_TOLERANCE_ZH[profile.risk_tolerance]}画像的常规水平。"
            ),
            "evidence_ref": "position_gap.gap_pct",
        })
    h21 = (ctx.get("vol_bands") or {}).get("h21") or {}
    if profile.horizon == "short" and h21:
        width = float(h21.get("p90", 0.0)) - float(h21.get("p10", 0.0))
        if width > SHORT_HORIZON_BAND_WIDTH:
            risk_flags.append({
                "flag": "short_horizon_high_vol",
                "detail": (
                    f"您的期限画像为短期，而未来 21 天收益分布带宽达 {width:.0%}，"
                    "短期不确定性高于常态。"
                ),
                "evidence_ref": "vol_bands.h21",
            })
    dev_z = (ctx.get("fair_value") or {}).get("deviation_z")
    if dev_z is not None and abs(float(dev_z)) >= STRUCTURAL_Z:
        risk_flags.append({
            "flag": "structural_valuation_deviation",
            "detail": (
                f"金价相对宏观公允锚的偏离达 {float(dev_z):+.1f}σ，处于结构性偏离期，"
                "长期估值锚的可靠性下降。"
            ),
            "evidence_ref": "fair_value.deviation_z",
        })
    if ctx.get("data_stale"):
        risk_flags.append({
            "flag": "stale_data",
            "detail": (
                f"底层数据截至 {ctx.get('data_asof')}（{ctx.get('data_age_days')} 天前），"
                "已超出新鲜度阈值，结论时效性受限。"
            ),
            "evidence_ref": "data_asof/data_age_days",
        })

    # ---- Advanced-layer rules (only when the optional fields are given) ----
    if profile.max_drawdown_pct is not None and h21:
        # Worst-decile 21-day move applied to the current position: if that
        # single bad month alone can breach the stated tolerance, say so.
        p10 = float(h21.get("p10", 0.0))
        potential_hit_pct = abs(min(p10, 0.0)) * float(profile.current_gold_pct)
        if potential_hit_pct > float(profile.max_drawdown_pct):
            risk_flags.append({
                "flag": "drawdown_tolerance_mismatch",
                "detail": (
                    f"未来 21 天最差十分位金价变动（{p10:+.1%}）作用于当前仓位 "
                    f"{profile.current_gold_pct:.0f}%，对组合的潜在冲击约 "
                    f"{potential_hit_pct:.1f}%，已超过您声明的最大回撤承受力 "
                    f"{profile.max_drawdown_pct:.0f}%。"
                ),
                "evidence_ref": "vol_bands.h21.p10 x profile.current_gold_pct",
            })
    if profile.leverage_attitude in ("medium", "high"):
        risk_flags.append({
            "flag": "leverage_out_of_scope",
            "detail": (
                "本研究口径的参考区间均以无杠杆现货敞口计算；"
                "使用杠杆会成倍放大区间外风险，且不在本系统的评估范围内。"
            ),
            "evidence_ref": "profile.leverage_attitude",
        })
    if profile.liquidity_need == "high" and profile.horizon == "long":
        risk_flags.append({
            "flag": "liquidity_horizon_mismatch",
            "detail": (
                "您声明的流动性需求为高，但期限画像为长期——"
                "长期视角的估值锚回归可能需要数月甚至更久，两者存在结构性矛盾。"
            ),
            "evidence_ref": "profile.liquidity_need vs profile.horizon",
        })

    # Horizon-matched evidence: the view-book section for this profile.
    section_key = _HORIZON_TO_SECTION[profile.horizon]
    horizon_evidence = {
        "horizon": section_key,
        "section": book.get(section_key),
        "evidence_ref": f"market_view.{section_key}",
    }

    return {
        "profile": profile.model_dump(),
        "reference_range": reference_range,
        "position_gap": position_gap,
        "risk_flags": risk_flags,
        "horizon_evidence": horizon_evidence,
        "meta": book.get("meta", {}),
        "degraded": degraded,
        "disclaimer": ALLOCATION_DISCLAIMER,
    }


# --------------------------------------------------------------------------- #
def draft_personal_narrative(
    facts: Dict[str, Any],
    profile: InvestorProfile,
) -> PersonalNarrative:
    """Deterministic Chinese narrative: the offline path and the fallback."""
    tol_zh = _RISK_TOLERANCE_ZH[profile.risk_tolerance]
    hor_zh = _HORIZON_ZH[profile.horizon]

    overview = (
        f"以下内容基于您提供的画像（{tol_zh}型 · {hor_zh}视角 · "
        f"当前黄金仓位 {profile.current_gold_pct:.0f}%）与系统当前研究结论整理，"
        "仅为教育型研究参考。"
    )

    rng = facts.get("reference_range") or {}
    gap = facts.get("position_gap") or {}
    if rng.get("available"):
        lo, hi = rng["range_pct"]
        explain = (
            "（参考区间指该画像下研究口径的常规配置范围，非任何操作指令）"
            if profile.experience == "novice" else ""
        )
        status = gap.get("status")
        if status == "within":
            gap_txt = f"您的当前仓位处于该参考区间内。"
        elif status == "above":
            gap_txt = f"您的当前仓位高于该参考区间上沿约 {gap['gap_pct']:.1f} 个百分点。"
        elif status == "below":
            gap_txt = f"您的当前仓位低于该参考区间下沿约 {gap['gap_pct']:.1f} 个百分点。"
        else:
            gap_txt = "仓位差距无法计算。"
        position_analysis = (
            f"{tol_zh}画像的研究参考区间为组合的 {lo:.1f}%–{hi:.1f}%{explain}。{gap_txt}"
        )
    else:
        position_analysis = "配置参考区间当前不可用（数据降级），本次不提供仓位对比。"

    risk_notes = [f["detail"] for f in facts.get("risk_flags", [])] or [
        "当前未触发结构化风险提示规则。"
    ]

    section = (facts.get("horizon_evidence") or {}).get("section") or {}
    if section.get("available"):
        horizon_note = f"与您{hor_zh}视角对应的系统观点：{section['core_view']}"
    else:
        horizon_note = f"{hor_zh}观点当前数据降级，不提供该节结论。"

    return PersonalNarrative(
        overview=overview,
        position_analysis=position_analysis,
        risk_notes=risk_notes,
        horizon_note=horizon_note,
        disclaimer=ALLOCATION_DISCLAIMER,
    )
