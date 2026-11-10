"""Tests for the immutable weekly signal ledger and its shadow-portfolio scoring."""
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import pytest

from signal_ledger import (
    JsonlLedgerStore,
    LedgerIntegrityError,
    LedgerUnavailableError,
    MemoryLedgerStore,
    PostgresLedgerStore,
    StalePublicationDataError,
    build_ledger_store,
    build_publication,
    publication_id_for,
    publish_weekly,
    score_track_record,
    verify_publication,
)


def _normalized_sql(value: str) -> str:
    return " ".join(value.lower().split())


class _FakeCursor:
    def __init__(self, adapter):
        self._adapter = adapter
        self._result = []
        self.closed = False
        self.rowcount = -1

    def execute(self, sql, params=None):
        normalized = _normalized_sql(sql)
        self._adapter.statements.append((normalized, params))
        if self._adapter.fail_on and self._adapter.fail_on in normalized:
            raise RuntimeError("adapter failure containing a secret DSN")
        if normalized.startswith("insert into signal_publications"):
            publication_id, published_at, payload = params
            record = json.loads(payload)
            with self._adapter.lock:
                inserted = publication_id not in self._adapter.rows
                self._adapter.rows.setdefault(
                    publication_id, (publication_id, published_at, record)
                )
                self.rowcount = 1 if inserted else 0
        elif normalized.startswith("select publication_id"):
            barrier = None
            with self._adapter.lock:
                self._result = sorted(
                    self._adapter.rows.values(), key=lambda row: row[1]
                )
                self._adapter.select_calls += 1
                if self._adapter.select_calls <= 2:
                    barrier = self._adapter.initial_select_barrier
            if barrier is not None:
                barrier.wait(timeout=2)

    def fetchall(self):
        return list(self._result)

    def close(self):
        self.closed = True


class _FakeConnection:
    def __init__(self, adapter):
        self._adapter = adapter
        self.cursors = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        cursor = _FakeCursor(self._adapter)
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class _FakePostgresAdapter:
    def __init__(self):
        self.connections = []
        self.connect_kwargs = []
        self.statements = []
        self.rows = {}
        self.lock = threading.Lock()
        self.fail_on = None
        self.initial_select_barrier = None
        self.select_calls = 0

    def connect(self, database_url, **kwargs):
        assert database_url == "postgresql://ignored-in-tests"
        self.connect_kwargs.append(kwargs)
        connection = _FakeConnection(self)
        self.connections.append(connection)
        return connection


def _postgres_store(adapter: _FakePostgresAdapter) -> PostgresLedgerStore:
    return PostgresLedgerStore(
        "postgresql://ignored-in-tests",
        connect=adapter.connect,
        jsonb_wrapper=lambda payload: json.dumps(
            payload, ensure_ascii=False, sort_keys=True
        ),
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


@pytest.mark.parametrize("stale_fields", [{"data_stale": True}, {"degraded": {"market_data": "unavailable"}}])
def test_stale_context_only_allows_existing_publication(stale_fields):
    store = MemoryLedgerStore()
    stale_ctx = {**_ctx(), **stale_fields}

    with pytest.raises(StalePublicationDataError, match="stale or unavailable"):
        publish_weekly(store, stale_ctx, now=MON)
    assert store.load_all() == []

    first, _created = publish_weekly(store, _ctx(), now=MON)
    assert publish_weekly(store, stale_ctx, now=MON) == (first, False)


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


def test_postgres_store_initializes_exact_append_only_schema():
    adapter = _FakePostgresAdapter()

    _postgres_store(adapter)

    assert adapter.statements == [
        (
            "create table if not exists signal_publications( publication_id text primary key, published_at timestamptz not null, payload jsonb not null )",
            None,
        )
    ]
    connection = adapter.connections[0]
    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert connection.closed is True
    assert all(cursor.closed for cursor in connection.cursors)


def test_postgres_connections_enforce_a_bounded_timeout():
    adapter = _FakePostgresAdapter()

    PostgresLedgerStore(
        "postgresql://ignored-in-tests",
        connect=adapter.connect,
        connect_timeout_seconds=4,
        jsonb_wrapper=lambda payload: json.dumps(payload),
    )

    assert adapter.connect_kwargs == [
        {
            "connect_timeout": 4,
            "options": (
                "-c statement_timeout=4000 "
                "-c lock_timeout=4000 "
                "-c idle_in_transaction_session_timeout=4000"
            ),
            "tcp_user_timeout": 4000,
        }
    ]


def test_postgres_healthcheck_executes_select_one():
    adapter = _FakePostgresAdapter()
    store = _postgres_store(adapter)

    assert store.healthcheck() is True
    assert adapter.statements[-1] == ("select 1", None)


def test_store_factory_uses_jsonl_only_in_explicit_development(tmp_path):
    selected = build_ledger_store(
        app_env="development",
        database_url=None,
        jsonl_path=tmp_path / "ledger.jsonl",
    )

    assert isinstance(selected, JsonlLedgerStore)


def test_store_factory_requires_postgres_outside_development():
    with pytest.raises(LedgerUnavailableError):
        build_ledger_store(
            app_env="production",
            database_url=None,
            jsonl_path="ignored.jsonl",
        )


def test_store_factory_selects_injected_postgres_factory():
    selected = MemoryLedgerStore()
    calls = []

    result = build_ledger_store(
        app_env="demo",
        database_url="postgresql://ledger.example/app",
        jsonl_path="ignored.jsonl",
        connect_timeout_seconds=3,
        postgres_store_factory=lambda url, **kwargs: calls.append((url, kwargs))
        or selected,
    )

    assert result is selected
    assert calls == [
        ("postgresql://ledger.example/app", {"connect_timeout_seconds": 3})
    ]


def test_postgres_append_uses_conflict_safe_insert_and_jsonb_serialization():
    adapter = _FakePostgresAdapter()
    store = _postgres_store(adapter)
    record = build_publication(_ctx(), now=MON)

    store.append(record)
    store.append(record)

    inserts = [
        (sql, params)
        for sql, params in adapter.statements
        if sql.startswith("insert into signal_publications")
    ]
    assert len(inserts) == 2
    assert all(
        "on conflict (publication_id) do nothing" in sql for sql, _params in inserts
    )
    assert json.loads(inserts[0][1][2]) == record
    assert list(adapter.rows) == ["2026-W29"]
    assert all(connection.commits == 1 for connection in adapter.connections)
    assert all(connection.closed for connection in adapter.connections)
    assert all(
        cursor.closed
        for connection in adapter.connections
        for cursor in connection.cursors
    )


def test_postgres_load_all_returns_validated_payloads_ordered_by_published_at():
    adapter = _FakePostgresAdapter()
    store = _postgres_store(adapter)
    later = build_publication(_ctx(), now=NEXT_MON)
    earlier = build_publication(_ctx(), now=MON)
    store.append(later)
    store.append(earlier)

    records = store.load_all()

    assert [record["publication_id"] for record in records] == [
        "2026-W29",
        "2026-W30",
    ]
    select_sql = [
        sql for sql, _params in adapter.statements if sql.startswith("select publication_id")
    ]
    assert select_sql == [
        "select publication_id, published_at, payload from signal_publications order by published_at asc"
    ]
    load_connection = adapter.connections[-1]
    assert load_connection.commits == 1
    assert load_connection.closed is True
    assert all(cursor.closed for cursor in load_connection.cursors)


@pytest.mark.parametrize(
    "bad_payload",
    [
        "not-json",
        {"publication_id": "2026-W29", "published_at": MON.isoformat()},
        {
            **build_publication(_ctx(), now=MON),
            "publication_id": "different-id",
        },
    ],
)
def test_postgres_load_all_rejects_malformed_or_tampered_rows(bad_payload):
    adapter = _FakePostgresAdapter()
    store = _postgres_store(adapter)
    adapter.rows["2026-W29"] = ("2026-W29", MON.isoformat(), bad_payload)

    with pytest.raises(LedgerIntegrityError):
        store.load_all()

    connection = adapter.connections[-1]
    assert connection.rollbacks == 1
    assert connection.closed is True
    assert all(cursor.closed for cursor in connection.cursors)


def test_postgres_transaction_failure_rolls_back_and_closes_resources():
    adapter = _FakePostgresAdapter()
    store = _postgres_store(adapter)
    adapter.fail_on = "insert into signal_publications"

    with pytest.raises(LedgerUnavailableError) as exc_info:
        store.append(build_publication(_ctx(), now=MON))

    formatted = "".join(
        traceback.format_exception(
            exc_info.type, exc_info.value, exc_info.tb, chain=True
        )
    )
    assert "secret DSN" not in str(exc_info.value)
    assert "secret DSN" not in formatted
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None

    connection = adapter.connections[-1]
    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert connection.closed is True
    assert all(cursor.closed for cursor in connection.cursors)


def test_postgres_concurrent_append_keeps_one_publication_per_id():
    adapter = _FakePostgresAdapter()
    store = _postgres_store(adapter)
    record = build_publication(_ctx(), now=MON)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(store.append, record) for _ in range(2)]
        for future in futures:
            future.result()

    assert list(adapter.rows) == ["2026-W29"]
    assert store.load_all() == [record]


def test_postgres_concurrent_publish_reports_one_creator_and_one_existing_record():
    adapter = _FakePostgresAdapter()
    adapter.initial_select_barrier = threading.Barrier(2)
    store = _postgres_store(adapter)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish_weekly, store, _ctx(), now=MON),
            executor.submit(
                publish_weekly,
                store,
                _ctx(),
                now=MON + pd.Timedelta(hours=1),
            ),
        ]
        results = [future.result() for future in futures]

    assert sum(created for _record, created in results) == 1
    assert results[0][0] == results[1][0] == store.load_all()[0]
    assert list(adapter.rows) == ["2026-W29"]


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
