"""Tests for the investor profile model and the personal research rule engine."""
import pytest
from pydantic import ValidationError

from investor_profile import InvestorProfile, PersonalNarrative
from personal_research import (
    build_personal_facts,
    check_no_directive_language,
    draft_personal_narrative,
)


# --------------------------------------------------------------------------- #
# Profile model
# --------------------------------------------------------------------------- #
def test_profile_valid():
    p = InvestorProfile(
        risk_tolerance="balanced", horizon="mid",
        current_gold_pct=10.0, experience="novice",
    )
    assert p.risk_tolerance == "balanced"


@pytest.mark.parametrize("field,value", [
    ("risk_tolerance", "yolo"),
    ("horizon", "forever"),
    ("current_gold_pct", -1.0),
    ("current_gold_pct", 101.0),
    ("experience", "guru"),
])
def test_profile_rejects_invalid(field, value):
    payload = {
        "risk_tolerance": "balanced", "horizon": "mid",
        "current_gold_pct": 10.0, "experience": "novice",
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        InvestorProfile(**payload)


def test_profile_rejects_extra_fields():
    with pytest.raises(ValidationError):
        InvestorProfile(
            risk_tolerance="balanced", horizon="mid",
            current_gold_pct=10.0, experience="novice", account_no="123",
        )


# --------------------------------------------------------------------------- #
# Rule engine
# --------------------------------------------------------------------------- #
def _ctx(*, stress=0.05, band_width=0.06, deviation_z=0.9, stale=False) -> dict:
    half = band_width / 2.0
    return {
        "data_asof": "2026-07-14",
        "data_age_days": 2,
        "data_stale": stale,
        "is_realtime": False,
        "degraded": {},
        "vol_bands": {
            "h1": {"p10": -0.011, "p50": 0.0, "p90": 0.011,
                   "ann_vol_forecast": 0.14, "horizon_days": 1.0},
            "h5": {"p10": -0.023, "p50": 0.0, "p90": 0.024,
                   "ann_vol_forecast": 0.14, "horizon_days": 5.0},
            "h21": {"p10": -half, "p50": 0.001, "p90": half,
                    "ann_vol_forecast": 0.14, "horizon_days": 21.0},
        },
        "regime_posterior": {
            "latest": {"calm": 1.0 - 0.2 - stress, "elevated": 0.2, "stress": stress},
        },
        "macro_factors": {
            "factors_used": ["real_rate", "usd"],
            "latest": {"real_rate": 0.8, "usd": 0.6},
            "composite": 0.6,
        },
        "fair_value": {"deviation_pct": 12.0, "deviation_z": deviation_z,
                       "regime_break": False},
        "scenario_cone": {"checkpoints": {"d90": {"p10": 3700.0, "p50": 4025.0,
                                                  "p90": 4400.0, "prob_above_spot": 0.54}}},
        "allocation": {
            "conservative": {"reference_range_pct": [2.0, 8.0]},
            "balanced": {"reference_range_pct": [5.0, 12.0]},
            "aggressive": {"reference_range_pct": [8.0, 18.0]},
        },
    }


def _profile(**over) -> InvestorProfile:
    base = dict(risk_tolerance="balanced", horizon="mid",
                current_gold_pct=8.0, experience="experienced")
    base.update(over)
    return InvestorProfile(**base)


def test_gap_within_range():
    facts = build_personal_facts(_profile(current_gold_pct=8.0), _ctx())
    assert facts["position_gap"]["status"] == "within"
    assert facts["reference_range"]["range_pct"] == [5.0, 12.0]


def test_gap_below_and_above():
    below = build_personal_facts(_profile(current_gold_pct=2.0), _ctx())
    assert below["position_gap"]["status"] == "below"
    assert below["position_gap"]["gap_pct"] == pytest.approx(3.0)

    above = build_personal_facts(_profile(current_gold_pct=15.0), _ctx())
    assert above["position_gap"]["status"] == "above"
    assert above["position_gap"]["gap_pct"] == pytest.approx(3.0)


def test_risk_flag_above_range_in_stress():
    facts = build_personal_facts(
        _profile(current_gold_pct=15.0), _ctx(stress=0.40),
    )
    flags = {f["flag"] for f in facts["risk_flags"]}
    assert "position_above_range_in_stress" in flags

    calm = build_personal_facts(_profile(current_gold_pct=15.0), _ctx(stress=0.05))
    assert "position_above_range_in_stress" not in {f["flag"] for f in calm["risk_flags"]}


def test_risk_flag_far_above_range():
    facts = build_personal_facts(_profile(current_gold_pct=18.0), _ctx())
    assert "position_far_above_range" in {f["flag"] for f in facts["risk_flags"]}


def test_risk_flag_short_horizon_high_vol():
    facts = build_personal_facts(
        _profile(horizon="short"), _ctx(band_width=0.12),
    )
    assert "short_horizon_high_vol" in {f["flag"] for f in facts["risk_flags"]}
    calm = build_personal_facts(_profile(horizon="short"), _ctx(band_width=0.05))
    assert "short_horizon_high_vol" not in {f["flag"] for f in calm["risk_flags"]}


def test_risk_flag_structural_deviation_and_stale():
    facts = build_personal_facts(
        _profile(), _ctx(deviation_z=2.4, stale=True),
    )
    flags = {f["flag"] for f in facts["risk_flags"]}
    assert "structural_valuation_deviation" in flags
    assert "stale_data" in flags


def test_horizon_evidence_matches_profile_horizon():
    short = build_personal_facts(_profile(horizon="short"), _ctx())
    assert short["horizon_evidence"]["horizon"] == "short_term"
    long_ = build_personal_facts(_profile(horizon="long"), _ctx())
    assert long_["horizon_evidence"]["horizon"] == "long_term"


def test_every_fact_block_has_evidence_ref():
    facts = build_personal_facts(_profile(), _ctx())
    assert facts["reference_range"]["evidence_ref"]
    assert facts["position_gap"]["evidence_ref"]
    for flag in facts["risk_flags"]:
        assert flag["evidence_ref"], flag["flag"]


def test_missing_allocation_degrades_explicitly():
    ctx = _ctx()
    del ctx["allocation"]
    facts = build_personal_facts(_profile(), ctx)
    assert facts["reference_range"]["available"] is False
    assert facts["degraded"]


# --------------------------------------------------------------------------- #
# Deterministic draft narrative
# --------------------------------------------------------------------------- #
def test_draft_narrative_contains_disclaimer_and_numbers():
    profile = _profile(current_gold_pct=15.0)
    facts = build_personal_facts(profile, _ctx(stress=0.40))
    draft = draft_personal_narrative(facts, profile)
    assert isinstance(draft, PersonalNarrative)
    assert draft.disclaimer
    assert "5" in draft.position_analysis and "12" in draft.position_analysis
    assert draft.risk_notes  # stress flag must surface in the notes


def test_draft_narrative_novice_gets_explanations():
    profile = _profile(experience="novice")
    facts = build_personal_facts(profile, _ctx())
    draft = draft_personal_narrative(facts, profile)
    combined = draft.overview + draft.horizon_note
    # Novice copy explains what the reference range means.
    assert "参考区间" in draft.position_analysis
    assert combined


# --------------------------------------------------------------------------- #
# Directive-language gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    "你应该买入黄金。",
    "建议卖出全部持仓。",
    "立即加仓到 20%。",
    "现在满仓黄金！",
    "赶紧清仓。",
    "必须买黄金抄底。",
])
def test_directive_language_caught(bad):
    ok, violations = check_no_directive_language([bad])
    assert ok is False
    assert violations


def test_compliant_language_passes():
    ok, violations = check_no_directive_language([
        "您的当前仓位 15% 高于 balanced 画像的参考区间 5%–12% 上沿约 3 个百分点。",
        "在压力状态概率 40% 的背景下，历史上高于区间的持仓回撤波动更大，供您参考。",
        "本内容为教育型研究参考，不构成投资建议。",
    ])
    assert ok is True
    assert violations == []
