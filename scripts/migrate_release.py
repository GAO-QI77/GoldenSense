"""Inspect or apply Golden Sense database changes using the configured DATABASE_URL."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LEGACY = """
coalesce(request_payload->>'feedback_token_hash', '') !~ '^[0-9a-f]{64}$'
and (request_payload->>'legacy_retention_version') is distinct from '1'
"""


def release_plan(conn: psycopg.Connection) -> dict:
    memory, news, traces = conn.execute("""
        select to_regclass('historical_events') is not null,
               to_regclass('news_captures') is not null,
               to_regclass('agent_analysis_traces') is not null
    """).fetchone()
    missing = []
    if memory:
        present = {
            row[0] for row in conn.execute("""
                select attname from pg_attribute
                where attrelid = to_regclass('historical_events') and not attisdropped
            """)
        }
        missing = sorted({"available_at", "gold_t1_available_at", "gold_t7_available_at"} - present)
    pending = conn.execute(
        f"select count(*) from agent_analysis_traces where {LEGACY}"
    ).fetchone()[0] if traces else 0
    return {
        "historical_events_table": memory,
        "memory_missing_columns": missing,
        "news_capture_table": news,
        "agent_trace_table": traces,
        "legacy_traces_pending": pending,
    }


def apply_schema(conn: psycopg.Connection) -> None:
    from memory_ingestion import _ensure_schema
    from news_ingest_service import ensure_news_capture_schema

    if conn.execute("select to_regclass('historical_events') is not null").fetchone()[0]:
        # Existing tables retain their embedding type and dimension.
        _ensure_schema(conn, embedding_dim=384)
    with conn.cursor() as cur:
        ensure_news_capture_schema(cur)
    conn.commit()


def redact_legacy_traces(conn: psycopg.Connection, *, batch_size: int) -> int:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if not conn.execute("select to_regclass('agent_analysis_traces') is not null").fetchone()[0]:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            f"""
            with pending as (
                select analysis_id, request_payload, response_payload
                from agent_analysis_traces where {LEGACY}
                order by analysis_id limit %s for update skip locked
            )
            update agent_analysis_traces as trace
            set request_payload = jsonb_build_object(
                    'legacy_retention_version', 1,
                    'horizon', case
                        when pending.request_payload->>'horizon' in ('short_term', 'mid_term', 'long_term', '24h', '7d', '30d')
                            then pending.request_payload->>'horizon'
                        when pending.response_payload #>> '{{summary_card,horizon}}' in ('short_term', 'mid_term', 'long_term', '24h', '7d', '30d')
                            then pending.response_payload #>> '{{summary_card,horizon}}'
                        else 'short_term' end
                ),
                tool_trace = '[]'::jsonb,
                evidence_payload = '{{}}'::jsonb,
                response_payload = case
                    when jsonb_typeof(pending.response_payload->'summary_card') = 'object'
                         and pending.response_payload->'summary_card' <> '{{}}'::jsonb
                    then jsonb_build_object('summary_card', jsonb_build_object(
                        'stance', case
                            when pending.response_payload #>> '{{summary_card,stance}}'
                                 in ('偏多', '偏空', '中性', '高风险观望')
                            then pending.response_payload #>> '{{summary_card,stance}}' else '中性' end,
                        'confidence_band', case
                            when pending.response_payload #>> '{{summary_card,confidence_band}}'
                                 in ('低', '中', '高')
                            then pending.response_payload #>> '{{summary_card,confidence_band}}' else '中' end
                    ))
                    else '{{}}'::jsonb end,
                feedback_rating = case when trace.feedback_rating in ('helpful', 'not_helpful')
                                       then trace.feedback_rating end,
                feedback_comment = null
            from pending
            where trace.analysis_id = pending.analysis_id
            """,
            (batch_size,),
        )
        changed = cur.rowcount
    conn.commit()
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--apply-schema", action="store_true", help="Apply additive schema changes")
    actions.add_argument("--redact-legacy", action="store_true", help="Permanently minimize old trace payloads")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        parser.error("DATABASE_URL is required")
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            conn.read_only = not (args.apply_schema or args.redact_legacy)
            before = release_plan(conn)
            if args.apply_schema:
                apply_schema(conn)
            redacted = 0
            if args.redact_legacy:
                while True:
                    changed = redact_legacy_traces(conn, batch_size=args.batch_size)
                    redacted += changed
                    if changed < args.batch_size:
                        break
            print(json.dumps({
                "before": before, "after": release_plan(conn), "legacy_traces_redacted": redacted,
            }))
    except psycopg.Error as exc:
        # Database connection exceptions may include connection details.
        print(f"Release database check failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
