"""Tests for the unified market view book assembled from the research context."""
import pytest

from market_view import build_market_view


def _full_context() -> dict:
    """A hand-built research-context dict with every block present."""
    return {
        "data_source": "extended",
        "data_asof": "2026-07-14",
        "data_age_days": 2,
        "data_stale": False,
        "is_realtime": False,
        "degraded": {},
        # forecast_return_bands output: RETURN-space quantiles (not prices).
        "vol_bands": {
            "h1": {"p10": -0.0110, "p50": 0.0002, "p90": 0.0112,
                   "ann_vol_forecast": 0.145, "horizon_days": 1.0},
            "h5": {"p10": -0.0230, "p50": 0.0006, "p90": 0.0245,
                   "ann_vol_forecast": 0.145, "horizon_days": 5.0},
            "h21": {"p10": -0.0350, "p50": 0.0015, "p90": 0.0380,
                    "ann_vol_forecast": 0.145, "horizon_days": 21.0},
        },
        "regime_posterior": {
            "latest": {"calm": 0.70, "elevated": 0.25, "stress": 0.05},
        },
        "macro_factors": {
            "factors_used": ["real_rate", "usd", "breakeven", "flow_confirmation"],
            "latest": {"real_rate": 0.8, "usd": 0.6, "breakeven": 0.55,
                       "flow_confirmation": 0.5},
            "composite": 0.6125,
        },
        "fair_value": {
            "deviation_pct": 12.0,
            "deviation_z": 0.9,
            "interpretation": "正常估值波动",
            "regime_break": False,
        },
        "scenario_cone": {
            "spot": 4000.0,
            "checkpoints": {
                "d30": {"p10": 3850.0, "p50": 4010.0, "p90": 4180.0,
                        "prob_above_spot": 0.55},
                "d90": {"p10": 3700.0, "p50": 4025.0, "p90": 4400.0,
                        "prob_above_spot": 0.54},
            },
        },
        "flagship": {
            "metrics": {"sharpe": 0.70, "max_drawdown": -0.21},
        },
        "allocation": {
            "balanced": {"reference_range_pct": [5.0, 12.0]},
        },
    }


def test_full_context_builds_three_horizons():
    book = build_market_view(_full_context())
    assert set(book) >= {"short_term", "mid_term", "long_term", "meta"}
    for horizon in ("short_term", "mid_term", "long_term"):
        section = book[horizon]
        assert section["available"] is True, horizon
        assert section["core_view"], horizon
        assert section["confidence"] in {"低", "中", "高"}, horizon
        assert section["evidence"], horizon
        assert section["invalidation"], horizon
    meta = book["meta"]
    assert meta["data_asof"] == "2026-07-14"
    assert meta["is_realtime"] is False
    assert meta["data_age_days"] == 2


def test_missing_blocks_degrade_explicitly():
    ctx = {
        "degraded": {
            "vol_bands": "boom",
            "regime_posterior": "boom",
            "macro_factors": "boom",
            "fair_value": "boom",
            "scenario_cone": "boom",
        },
        "data_asof": "2026-07-14",
        "is_realtime": False,
    }
    book = build_market_view(ctx)
    for horizon in ("short_term", "mid_term", "long_term"):
        section = book[horizon]
        assert section["available"] is False, horizon
        assert section["degraded_reason"], horizon
    # Degradation map is carried through untouched.
    assert book["meta"]["degraded"]["vol_bands"] == "boom"


def test_mid_term_confidence_tracks_dominant_state():
    ctx = _full_context()
    ctx["regime_posterior"]["latest"] = {"calm": 0.1, "elevated": 0.1, "stress": 0.8}
    book = build_market_view(ctx)
    mid = book["mid_term"]
    assert mid["confidence"] == "高"
    assert "压力" in mid["core_view"]

    ctx["regime_posterior"]["latest"] = {"calm": 0.4, "elevated": 0.35, "stress": 0.25}
    mid_low = build_market_view(ctx)["mid_term"]
    assert mid_low["confidence"] == "低"


def test_short_term_wide_band_lowers_confidence():
    ctx = _full_context()
    # 21-day return band width p90 - p10 = 15% > 8% threshold -> low confidence.
    ctx["vol_bands"]["h21"] = {"p10": -0.075, "p50": 0.0, "p90": 0.075,
                               "ann_vol_forecast": 0.30, "horizon_days": 21.0}
    book = build_market_view(ctx)
    assert book["short_term"]["confidence"] == "低"

    # Narrow band -> medium confidence.
    ctx["vol_bands"]["h21"] = {"p10": -0.025, "p50": 0.001, "p90": 0.028,
                               "ann_vol_forecast": 0.12, "horizon_days": 21.0}
    assert build_market_view(ctx)["short_term"]["confidence"] == "中"


def test_long_term_structural_deviation_flagged():
    ctx = _full_context()
    ctx["fair_value"]["deviation_z"] = 2.5
    ctx["fair_value"]["deviation_pct"] = 55.0
    book = build_market_view(ctx)
    long_term = book["long_term"]
    assert "结构性" in long_term["core_view"]
    # Invalidation conditions must mention the z-score threshold.
    assert any("z" in inv or "±2" in inv for inv in long_term["invalidation"])
