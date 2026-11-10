"""Tests for the immutable weekly signal ledger and its shadow-portfolio scoring."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from signal_ledger import (
    JsonlLedgerStore,
    MemoryLedgerStore,
    build_publication,
    publication_id_for,
    publish_weekly,
    score_track_record,
    verify_publication,
)


def _ctx() -> dict:
    return {
        "data_source": "extended",
        "data_asof": "2026-07-13",
        "degraded": {},
        "regime_posterior": {"latest": {"calm": 0.7, "elevated": 0.25, "stress": 0.05}},
        "fair_value": {"deviation_pct": 12.0, "deviation_z": 0.9},
        "macro_factors": {"composite": 0.61, "factors_used": ["real_rate"]},
        "vol_bands": {
            "h1": {"p10": -0.011, "p50": 0.0, "p90": 0.011,
                   "ann_vol_forecast": 0.14, "horizon_days": 1.0},
            "h5": {"p10": -0.023, "p50": 0.0, "p90": 0.024,
                   "ann_vol_forecast": 0.14, "horizon_days": 5.0},
            "h21": {"p10": -0.035, "p50": 0.001, "p90": 0.038,
                    "ann_vol_forecast": 0.14, "horizon_days": 21.0},
        },
        "allocation": {
            "conservative": {"reference_range_pct": [2.0, 8.0]},
            "balanced": {"reference_range_pct": [5.0, 12.0]},
            "aggressive": {"reference_range_pct": [8.0, 18.0]},
        },
    }


MON = datetime(2026, 7, 13, 9, 0, tzinfo=timezone.utc)   # Monday of ISO week 29
NEXT_MON = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)


def test_publication_id_iso_week():
    assert publication_id_for(MON) == "2026-W29"
    sunday = datetime(2026, 7, 19, 23, 0, tzinfo=timezone.utc)
    assert publication_id_for(sunday) == "2026-W29"
    assert publication_id_for(NEXT_MON) == "2026-W30"


def test_build_publication_has_hash_disclaimer_and_midpoints():
    record = build_publication(_ctx(), now=MON)
    assert record["publication_id"] == "2026-W29"
    assert record["content_hash"].startswith("sha256:")
    assert record["disclaimer"]
    balanced = record["allocations"]["balanced"]
    assert balanced["range_pct"] == [5.0, 12.0]
    assert balanced["midpoint"] == pytest.approx(8.5)
    assert record["evidence_snapshot"]["fair_value_deviation_pct"] == 12.0
    assert record["evidence_class"] == "live_forward"
    # Market view summary is frozen in, one line per horizon.
    assert set(record["market_view_summary"]) == {"short_term", "mid_term", "long_term"}


def test_verify_detects_tampering():
    record = build_publication(_ctx(), now=MON)
    assert verify_publication(record) is True
    tampered = dict(record)
    tampered["allocations"] = {
        **record["allocations"],
        "balanced": {"range_pct": [10.0, 20.0], "midpoint": 15.0},
    }
    assert verify_publication(tampered) is False


def test_publish_weekly_idempotent():
    store = MemoryLedgerStore()
    first, created_first = publish_weekly(store, _ctx(), now=MON)
    again, created_again = publish_weekly(store, _ctx(), now=MON + pd.Timedelta(days=2))
    assert created_first is True
    assert created_again is False
    assert again == first
    assert len(store.load_all()) == 1


def test_jsonl_store_append_only(tmp_path):
    path = tmp_path / "ledger.jsonl"
    store = JsonlLedgerStore(path)
    publish_weekly(store, _ctx(), now=MON)
    content_week1 = path.read_text()
    publish_weekly(store, _ctx(), now=NEXT_MON)
    publish_weekly(store, _ctx(), now=NEXT_MON)  # idempotent republish
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    # Week-1 bytes are untouched by later publishes.
    assert path.read_text().startswith(content_week1)
    # Reload from disk round-trips.
    reloaded = JsonlLedgerStore(path).load_all()
    assert [r["publication_id"] for r in reloaded] == ["2026-W29", "2026-W30"]


# --------------------------------------------------------------------------- #
# Track-record scoring
# --------------------------------------------------------------------------- #
def _daily_prices(start: str, days: int, step: float) -> pd.Series:
    idx = pd.bdate_range(start, periods=days)
    return pd.Series([100.0 + i * step for i in range(days)], index=idx)


def test_empty_ledger_scores_empty():
    result = score_track_record([], _daily_prices("2026-07-13", 30, 1.0))
    assert result["per_profile"]["balanced"]["weeks_scored"] == 0
    assert result["per_profile"]["balanced"]["cum_return"] is None
    assert result["matured_through"] is None
    assert result["disclaimer"]
    assert result["evidence_class"] == "simulated_forward"
    assert result["per_profile"]["balanced"]["evidence_class"] == "simulated_forward"


def test_single_immature_week_not_scored():
    record = build_publication(_ctx(), now=MON)
    # Prices end the same week -> nothing matured.
    prices = _daily_prices("2026-07-13", 3, 1.0)
    result = score_track_record([record], prices)
    assert result["per_profile"]["balanced"]["weeks_scored"] == 0


def test_two_weeks_scored_with_cost():
    rec1 = build_publication(_ctx(), now=MON)
    rec2 = build_publication(_ctx(), now=NEXT_MON)
    # 10 business days from Mon 2026-07-13: prices 100, 102, ..., 118.
    prices = _daily_prices("2026-07-13", 10, 2.0)
    result = score_track_record([rec1, rec2], prices, cost_bps=5.0)
    prof = result["per_profile"]["balanced"]

    # Only date-level prices are available, so execution is at the next
    # session's close after each publication (102 -> 112).
    assert prof["weeks_scored"] == 1
    weight = 8.5 / 100.0
    gold_ret_week = 112.0 / 102.0 - 1.0
    fee = weight * 0.0005 / (1.0 + weight * 0.0005)
    expected = (1.0 + weight * gold_ret_week) * (1.0 - fee) - 1.0
    # Output is rounded to 6 decimals for display; tolerance matches.
    assert prof["cum_return"] == pytest.approx(expected, abs=5e-7)

    # Benchmark: 100% gold buy-and-hold over the same matured span.
    bench = result["benchmarks"]["gold_buy_hold"]
    assert bench["cum_return"] == pytest.approx(gold_ret_week, abs=5e-7)
    assert result["matured_through"] == "2026-W30"
    assert prof["evidence_class"] == "simulated_forward"
    assert bench["evidence_class"] == "simulated_forward"


def test_track_record_rebalances_from_drift_and_ignores_unavailable_prices():
    records = [build_publication(_ctx(), now=MON + pd.Timedelta(weeks=i)) for i in range(3)]
    prices = _daily_prices("2026-07-13", 15, 5.0)
    score = score_track_record(records, prices, cost_bps=100, decision_at=MON + pd.Timedelta(days=22))
    balanced = score["per_profile"]["balanced"]
    assert balanced["weeks_scored"] == 2
    weight = 0.085
    r1 = 130 / 105 - 1
    r2 = 155 / 130 - 1
    first_cost = weight * 0.01 / (1 + weight * 0.01)
    drift = weight * (1 + r1) / (1 + weight * r1)
    second_cost = (drift - weight) * 0.01 / (1 - weight * 0.01)
    expected = (1 - first_cost) * (1 + weight * r1) * (1 - second_cost) * (1 + weight * r2) - 1
    assert balanced["cum_return"] == pytest.approx(expected, abs=5e-7)
    early = score_track_record(records, prices, decision_at=MON + pd.Timedelta(days=15))
    assert early["per_profile"]["balanced"]["weeks_scored"] == 1


def test_signal_publication_is_idempotent_across_concurrent_store_instances(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "shared.jsonl"
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(
            lambda _: publish_weekly(JsonlLedgerStore(path), _ctx(), now=MON),
            range(16),
        ))
    assert sum(created for _, created in outcomes) == 1
    assert len(JsonlLedgerStore(path).load_all()) == 1


def test_verified_close_availability_controls_shadow_scoring(monkeypatch, tmp_path):
    from signal_ledger import score_configured_track_record

    records = [build_publication(_ctx(), now=MON + pd.Timedelta(weeks=i)) for i in range(2)]
    closes = tmp_path / "prices.csv"
    sessions = tmp_path / "sessions.csv"
    closes.write_text("Date,Gold\n2026-07-13,100\n2026-07-20,110\n")
    sessions.write_text(
        "Date,close_at,available_at\n"
        "2026-07-13,2026-07-13T18:00:00Z,2026-07-13T19:00:00Z\n"
        "2026-07-20,2026-07-20T18:00:00Z,2026-07-20T19:00:00Z\n"
    )
    monkeypatch.setenv("GOLD_VERIFIED_CLOSES_PATH", str(closes))
    monkeypatch.setenv("GOLD_VERIFIED_SESSIONS_PATH", str(sessions))
    early = score_configured_track_record(records, decision_at=NEXT_MON.replace(hour=18, minute=30))
    late = score_configured_track_record(records, decision_at=NEXT_MON.replace(hour=19))
    assert early["per_profile"]["balanced"]["weeks_scored"] == 0
    assert late["per_profile"]["balanced"]["weeks_scored"] == 1
    assert late["price_evidence"] == "session_timestamps"
    assert late["benchmarks"]["gold_buy_hold"]["cum_return"] == pytest.approx(0.10)

    monkeypatch.delenv("GOLD_VERIFIED_SESSIONS_PATH")
    with pytest.raises(ValueError, match="both"):
        score_configured_track_record(records, decision_at=NEXT_MON)


def test_static_comparator_uses_same_costs_and_initial_exposure():
    records = [build_publication(_ctx(), now=MON + pd.Timedelta(weeks=i)) for i in range(3)]
    score = score_track_record(records, _daily_prices("2026-07-13", 15, 5.0), cost_bps=100)
    benchmark = score["benchmarks"]["balanced_fixed_net"]
    assert benchmark["weeks_scored"] == 2
    assert benchmark["cost_bps"] == 100
    assert benchmark["evidence_class"] == "simulated_forward"
    assert benchmark["cum_return"] == score["per_profile"]["balanced"]["cum_return"]
