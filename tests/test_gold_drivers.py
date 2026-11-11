import numpy as np
import pandas as pd
import pytest

from gold_drivers import build_gold_drivers
from market_view import build_market_view


def test_gold_drivers_are_explanatory_and_use_visible_versions(tmp_path):
    rng = np.random.default_rng(12)
    dates = pd.bdate_range("2023-01-02", periods=520)
    rate_change = rng.normal(0, 0.03, len(dates))
    usd_change = rng.normal(0, 0.005, len(dates))
    oil_change = rng.normal(0, 0.015, len(dates))
    beta = np.where(np.arange(len(dates)) < len(dates) - 126, -0.10, -0.03)
    gold_change = beta * rate_change - 0.5 * usd_change
    raw = pd.DataFrame({
        "Gold": 2000 * np.exp(gold_change.cumsum()),
        "Real_10Y": 2 + rate_change.cumsum(),
        "USD_Index": 100 * np.exp(usd_change.cumsum()),
        "Crude_Oil": 70 * np.exp(oil_change.cumsum()),
    }, index=dates)
    day = dates[-1].date()
    path = tmp_path / "flows.csv"
    path.write_text(
        "series,observation_date,available_at,value\n"
        f"etf_net_tonnes,{day},{day}T14:00:00Z,2\n"
        f"etf_net_tonnes,{day},{day}T15:00:00Z,4\n"
        f"cot_managed_net,{day},{day}T15:00:00-12:00,9000\n"
        f"cot_other_net,{day},{day},8000\n"
    )
    out = build_gold_drivers(raw, flow_path=path)
    assert out["oil_rate_conditions"]["interpretation"] == "same_session_description"
    assert sum(row["n"] for row in out["oil_rate_conditions"]["groups"]) == 504
    assert out["rate_sensitivity"]["prior_log_pct_per_100bp"] == pytest.approx(-10)
    assert out["rate_sensitivity"]["recent_log_pct_per_100bp"] == pytest.approx(-3)
    assert out["flows"]["etf_net_tonnes"]["value"] == 4
    assert out["flows"]["cot_managed_net"]["status"] == "unavailable"
    assert out["flows"]["cot_other_net"]["status"] == "unavailable"


def test_view_book_surfaces_driver_evidence_with_missing_flow_explicit():
    out = build_market_view({
        "vol_bands": {"h21": {"p10": -0.03, "p50": 0.0, "p90": 0.03}},
        "regime_posterior": {"latest": {"calm": 0.8}},
        "macro_factors": {"composite": 0.5},
        "fair_value": {"deviation_pct": 5, "deviation_z": 0.4},
        "driver_attribution": {"latest": {"dollar": 0.003, "real_rate": -0.002, "residual": 0.001}},
        "gold_drivers": {
            "flows": {"etf_net_tonnes": {"status": "unavailable"}},
            "rate_sensitivity": {"recent_log_pct_per_100bp": -3, "prior_log_pct_per_100bp": -10},
            "oil_rate_conditions": {"groups": [{"oil": "up", "real_rate": "up", "n": 40, "p10": -0.02}]},
        },
    })
    assert "40" in out["short_term"]["evidence"][0]
    assert "缺失" in out["mid_term"]["evidence"][0]
    assert "实际利率 -0.20" in out["mid_term"]["evidence"][1]
    assert "−" in out["long_term"]["evidence"][0] or "-3" in out["long_term"]["evidence"][0]


def test_cot_capture_records_revisions_only_at_capture_time(tmp_path):
    from datetime import datetime, timezone
    from gold_drivers import flow_snapshot_at
    from scripts.refresh_gold_driver_flows import merge_observations

    path = tmp_path / "flows.csv"
    one = datetime(2026, 11, 8, tzinfo=timezone.utc)
    two = datetime(2026, 11, 10, tzinfo=timezone.utc)
    row = {"series": "cot_managed_net", "observation_date": "2026-11-03", "value": 100}
    assert merge_observations(path, [row], captured_at=one) == 1
    assert merge_observations(path, [row], captured_at=two) == 0
    assert merge_observations(path, [{**row, "value": 200}], captured_at=two) == 1
    assert flow_snapshot_at(path, one)["cot_managed_net"]["value"] == 100
    assert flow_snapshot_at(path, two)["cot_managed_net"]["value"] == 200
