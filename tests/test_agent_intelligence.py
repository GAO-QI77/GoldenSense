"""Tests for the analyst committee, narrative critic, and outcome tracker."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from analyst_committee import (
    DISAGREEMENT_CONFLICT_THRESHOLD,
    build_committee,
    macro_analyst,
    news_analyst,
    technical_analyst,
)
from narrative_critic import collect_evidence_numbers, extract_numbers, verify_narrative
from outcome_tracker import (
    backfill_outcomes,
    calibration_summary,
    committee_weight_adjustments,
)


# ----------------------------- committee ---------------------------------- #
def _regime(trend_score: float, vol_state: str = "calm") -> dict:
    return {
        "trend_score": trend_score,
        "regime": "uptrend" if trend_score > 0.66 else "mixed",
        "vol_state": vol_state,
        "target_exposure_pct": 50.0,
        "sufficient_history": True,
    }


def test_technical_analyst_maps_trend_score_to_stance():
    bull = technical_analyst(_regime(1.0))
    bear = technical_analyst(_regime(0.0))
    assert bull.stance == 1.0 and bear.stance == -1.0
    missing = technical_analyst(None)
    assert missing.confidence <= 0.2


def test_macro_analyst_valuation_dampens_stance():
    factors = {"composite": 1.0, "factors_used": ["real_rate_momentum"], "latest": {}}
    no_valuation = macro_analyst(factors, None)
    rich = macro_analyst(factors, {"deviation_pct": 30.0})
    cheap = macro_analyst(factors, {"deviation_pct": -30.0})
    assert rich.stance < no_valuation.stance < cheap.stance


def test_news_analyst_vix_gate_caps_bullish_stance():
    calm = news_analyst(0.8, vix_value=15.0)
    stressed = news_analyst(0.8, vix_value=35.0)
    assert calm.stance > 0
    assert stressed.stance <= 0.0
    assert stressed.confidence < calm.confidence


def test_committee_agreement_vs_disagreement():
    agree = build_committee(
        regime=_regime(1.0),
        macro_factors={"composite": 1.0, "factors_used": ["a", "b", "c", "d"],
                       "latest": {"flow_confirmation": 1.0}},
        fair_value=None,
        news_sentiment=0.8,
        vix_value=15.0,
    )
    conflict = build_committee(
        regime=_regime(1.0),
        macro_factors={"composite": 0.0, "factors_used": ["a", "b", "c", "d"],
                       "latest": {"flow_confirmation": 0.0}},
        fair_value={"deviation_pct": 30.0},
        news_sentiment=-0.8,
        vix_value=15.0,
    )
    assert agree["fused_label"] == "偏多"
    assert agree["disagreement"] < conflict["disagreement"]
    assert conflict["has_material_disagreement"] == (
        conflict["disagreement"] >= DISAGREEMENT_CONFLICT_THRESHOLD
    )
    # Weights are a probability distribution (rounded to 4 decimals each).
    assert abs(sum(agree["weights"].values()) - 1.0) < 1e-3


def test_committee_stress_regime_shifts_weight_to_macro():
    kwargs = dict(
        macro_factors={"composite": 0.5, "factors_used": ["a"], "latest": {}},
        fair_value=None,
        news_sentiment=0.0,
        vix_value=None,
    )
    calm = build_committee(regime=_regime(0.5, "calm"), **kwargs)
    stress = build_committee(regime=_regime(0.5, "stress"), **kwargs)
    assert stress["weights"]["macro"] > calm["weights"]["macro"]
    assert stress["weights"]["technical"] < calm["weights"]["technical"]


# ------------------------------- critic ----------------------------------- #
def test_extract_numbers_handles_commas_and_percent():
    assert extract_numbers("金价 4,024.5，涨幅 1.2%") == [4024.5, 1.2]


def test_verify_narrative_grounded_passes():
    evidence = [{"latest_price": 4024.0, "probability": 0.62}]
    passed, report = verify_narrative(
        ["当前金价约 4024，上行概率 62%。"], evidence
    )
    assert passed, report
    assert report["violations"] == []


def test_verify_narrative_catches_hallucinated_number():
    evidence = [{"latest_price": 4024.0}]
    passed, report = verify_narrative(["目标价 5500。"], evidence)
    assert not passed
    assert report["violations"][0]["value"] == 5500.0


def test_verify_narrative_ignores_small_counts():
    passed, _ = verify_narrative(["我们综合了 3 条证据与 4 位分析师观点。"], [{}])
    assert passed


def test_collect_evidence_numbers_walks_nested_payloads():
    numbers = collect_evidence_numbers([{"a": [{"b": "价格 2100 美元"}, 62.5]}])
    assert 2100.0 in numbers and 62.5 in numbers


# --------------------------- outcome tracker ------------------------------ #
def _price_series(start: str, n: int, drift: float) -> pd.Series:
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(2000 * np.exp(np.arange(n) * drift), index=idx)


def _row(created_at: datetime, stance: str, horizon: str = "24h", conf: str = "中") -> dict:
    return {
        "analysis_id": f"a-{created_at.date()}",
        "created_at": created_at.isoformat(),
        "request_payload": {"horizon": horizon},
        "response_payload": {
            "summary_card": {"stance": stance, "confidence_band": conf, "horizon": horizon}
        },
    }


def test_backfill_outcomes_scores_matured_rows_only():
    prices = _price_series("2026-01-01", 60, drift=0.002)  # rising market
    early = _row(datetime(2026, 1, 10, tzinfo=timezone.utc), "偏多")
    # Last price is 2026-03-01, so a call placed on 03-01 has not matured.
    too_recent = _row(datetime(2026, 3, 1, tzinfo=timezone.utc), "偏多")
    outcomes = backfill_outcomes([early, too_recent], prices)
    assert len(outcomes) == 1
    assert outcomes[0]["realized_direction"] == 1


def test_calibration_summary_hit_rate_and_brier():
    prices = _price_series("2026-01-01", 90, drift=0.002)
    rows = [
        _row(datetime(2026, 1, 5, tzinfo=timezone.utc), "偏多", conf="高"),
        _row(datetime(2026, 1, 12, tzinfo=timezone.utc), "偏空", conf="中"),
        _row(datetime(2026, 1, 19, tzinfo=timezone.utc), "中性"),
    ]
    outcomes = backfill_outcomes(rows, prices)
    summary = calibration_summary(outcomes)
    assert summary["total_scored"] == 3
    assert summary["directional_calls"] == 2
    assert summary["neutral_or_gated"] == 1
    assert summary["hit_rate"] == 0.5  # 偏多 hit, 偏空 miss in a rising market
    assert 0.0 <= summary["brier_score"] <= 1.0
    assert summary["by_stance"]["偏多"]["hit_rate"] == 1.0
    assert summary["by_stance"]["偏空"]["hit_rate"] == 0.0


def test_weight_adjustment_requires_samples_and_is_capped():
    assert committee_weight_adjustments({"hit_rate": 0.9, "directional_calls": 5}) == {
        "fused_confidence_multiplier": 1.0,
        "basis": "insufficient_samples",
    }
    strong = committee_weight_adjustments({"hit_rate": 1.0, "directional_calls": 100})
    weak = committee_weight_adjustments({"hit_rate": 0.0, "directional_calls": 100})
    assert strong["fused_confidence_multiplier"] == 1.2
    assert weak["fused_confidence_multiplier"] == 0.8
