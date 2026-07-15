"""Tests for the champion-challenger performance governance policy."""
from model_governance import (
    DEMOTE_BELOW,
    apply_confidence_derate,
    evaluate_governance,
)


def _summary(hit_rate, n):
    return {"hit_rate": hit_rate, "directional_calls": n}


def test_insufficient_data_keeps_champion():
    v = evaluate_governance(_summary(0.2, 5))
    assert v.mode == "insufficient_data"
    assert v.demoted is False
    assert v.confidence_multiplier == 1.0
    assert v.force_conservative is False


def test_none_hit_rate_is_insufficient():
    v = evaluate_governance({"hit_rate": None, "directional_calls": 100})
    assert v.mode == "insufficient_data"
    assert v.demoted is False


def test_strong_hit_rate_stays_champion_and_boosts_confidence():
    v = evaluate_governance(_summary(0.70, 60))
    assert v.mode == "champion"
    assert v.demoted is False
    assert v.confidence_multiplier > 1.0
    assert v.confidence_multiplier <= 1.20


def test_below_coin_flip_enters_watch_without_demotion():
    v = evaluate_governance(_summary(0.48, 40))
    assert v.mode == "watch"
    assert v.demoted is False
    assert v.confidence_multiplier < 1.0
    assert v.force_conservative is False


def test_sustained_poor_hit_rate_demotes():
    v = evaluate_governance(_summary(0.30, 50))
    assert v.mode == "demoted"
    assert v.demoted is True
    assert v.force_conservative is True
    # edge = (0.30-0.5)*2 = -0.4 -> multiplier = 1 - 0.4*0.20 = 0.92
    assert abs(v.confidence_multiplier - 0.92) < 1e-9
    assert f"{DEMOTE_BELOW:.0%}" in v.reason


def test_multiplier_is_bounded_both_directions():
    hi = evaluate_governance(_summary(1.0, 100)).confidence_multiplier
    lo = evaluate_governance(_summary(0.0, 100)).confidence_multiplier
    assert hi == 1.20
    assert lo == 0.80


def test_confidence_derate_only_when_forced():
    demoted = evaluate_governance(_summary(0.30, 50))
    champion = evaluate_governance(_summary(0.70, 50))
    assert apply_confidence_derate("高", demoted) == "中"
    assert apply_confidence_derate("中", demoted) == "低"
    assert apply_confidence_derate("低", demoted) == "低"
    assert apply_confidence_derate("高", champion) == "高"


def test_verdict_is_jsonable():
    import json

    payload = evaluate_governance(_summary(0.30, 50)).as_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["demoted"] is True
    assert "mode" in payload
