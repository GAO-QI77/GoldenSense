"""Real PostgreSQL migration tests, enabled with GOLDEN_TEST_POSTGRES_DSN."""
from datetime import datetime, timedelta, timezone
import os
import uuid

import psycopg
from psycopg import sql
import pytest

from memory_ingestion import HistoricalEvent, _ensure_schema, _upsert_events
from memory_retriever import _query_array_cosine, _query_pgvector


@pytest.fixture
def database():
    dsn = os.environ.get("GOLDEN_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set GOLDEN_TEST_POSTGRES_DSN to a disposable PostgreSQL database")
    schema = sql.Identifier(f"gs_test_{uuid.uuid4().hex}")
    with psycopg.connect(dsn) as conn:
        conn.execute(sql.SQL("create schema {}").format(schema))
        conn.execute(sql.SQL("set search_path = {}, public").format(schema))
        conn.commit()
        try:
            yield conn
        finally:
            conn.rollback()
            conn.execute(sql.SQL("drop schema {} cascade").format(schema))
            conn.commit()


@pytest.mark.parametrize("storage", ["array", "pgvector"])
def test_legacy_schema_migrates_and_filters_maturity(database, storage):
    vector_available = database.execute(
        "select exists(select from pg_available_extensions where name = 'vector')"
    ).fetchone()[0]
    if storage == "pgvector" and not vector_available:
        pytest.skip("pgvector is unavailable")
    if vector_available:
        database.execute("create extension if not exists vector with schema public")
    vector = storage == "pgvector"
    database.execute(sql.SQL("""
        create table historical_events (
            event_id uuid primary key, event_date timestamptz not null,
            headline text not null, context_summary text not null,
            embedding {} not null,
            gold_t1_return double precision not null,
            gold_t7_return double precision not null
        )
    """).format(sql.SQL("vector(2)" if vector else "double precision[]")))
    cutoff = datetime.now(timezone.utc) + timedelta(days=1)
    event_date = cutoff - timedelta(days=30)
    database.execute(sql.SQL("""
        insert into historical_events values (%s, %s, 'Legacy', 'Unknown arrival', {}, 0.1, 0.2)
    """).format(sql.SQL("'[1,0]'" if vector else "ARRAY[1.0,0.0]")), (uuid.uuid4(), event_date))
    database.commit()

    assert _ensure_schema(database, 2) == storage
    assert _ensure_schema(database, 2) == storage
    event = HistoricalEvent(
        uuid.uuid4(), event_date, "Verified", "Captured", [1.0, 0.0],
        0.01, 0.07, cutoff - timedelta(days=2), cutoff + timedelta(days=2),
    )
    assert _upsert_events(database, storage, [event]) == 1
    query = _query_pgvector if vector else _query_array_cosine
    assert query(database, [1.0, 0.0], 3, cutoff - timedelta(days=2)) == []
    rows = query(database, [1.0, 0.0], 3, cutoff)
    assert [row["headline"] for row in rows] == ["Verified"]
    assert rows[0]["gold_t1_return"] == pytest.approx(0.01)
    assert rows[0]["gold_t7_return"] is None
    assert query(database, [1.0, 0.0], 3, cutoff + timedelta(days=2))[0]["gold_t7_return"] == pytest.approx(0.07)


def test_release_migration_preserves_calibration_and_scrubs_legacy_text(database):
    from psycopg.types.json import Jsonb
    from outcome_tracker import extract_analysis_record
    from scripts.migrate_release import apply_schema, redact_legacy_traces, release_plan

    database.execute("""
        create table agent_analysis_traces (
            analysis_id text primary key, request_payload jsonb not null, tool_trace jsonb not null,
            evidence_payload jsonb not null, response_payload jsonb not null, feedback_rating text,
            feedback_comment text, created_at timestamptz default now()
        )
    """)
    for name, request in (
        ("legacy", {"horizon": "mid_term", "question": "private"}),
        ("current", {"horizon": "7d", "feedback_token_hash": "a" * 64}),
    ):
        database.execute("""
            insert into agent_analysis_traces values (%s, %s, '[{"private":true}]', '{"private":true}',
                '{"summary_card":{"stance":"偏多","confidence_band":"中","thesis":"private"}}',
                'helpful', 'private', now())
        """, (name, Jsonb(request)))
    database.commit()

    assert release_plan(database)["legacy_traces_pending"] == 1
    assert not release_plan(database)["news_capture_table"]
    apply_schema(database)
    assert release_plan(database)["news_capture_table"]
    assert redact_legacy_traces(database, batch_size=1) == 1
    assert redact_legacy_traces(database, batch_size=1) == 0
    assert release_plan(database)["legacy_traces_pending"] == 0

    legacy = database.execute("""
        select analysis_id, request_payload, response_payload, created_at,
               tool_trace, evidence_payload, feedback_comment
        from agent_analysis_traces where analysis_id='legacy'
    """).fetchone()
    assert "private" not in str(legacy)
    score = extract_analysis_record(dict(zip(("analysis_id", "request_payload", "response_payload", "created_at"), legacy)))
    assert (score["horizon"], score["stance"], score["confidence_band"]) == ("mid_term", "偏多", "中")
    assert database.execute("select feedback_comment from agent_analysis_traces where analysis_id='current'").fetchone()[0] == "private"
