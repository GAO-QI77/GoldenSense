from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import psycopg
from sentence_transformers import SentenceTransformer

from data_loader import market_availability_times
from memory_retriever import _detect_storage


@dataclass(frozen=True)
class HistoricalEvent:
    event_id: uuid.UUID
    event_date: datetime
    headline: str
    context_summary: str
    embedding: List[float]
    gold_t1_return: Optional[float]
    gold_t7_return: Optional[float]
    gold_t1_available_at: Optional[datetime]
    gold_t7_available_at: Optional[datetime]


def _vector_to_pgvector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def _load_market_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df["Gold"] = df["Gold"].astype(float)
    return df


def _iter_news_jsonl(path: Path) -> Iterable[Tuple[datetime, str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            published_at = str(obj.get("published_at", "")).strip()
            title = str(obj.get("title", "")).strip()
            body = str(obj.get("body", "")).strip()
            if not published_at or not title:
                continue
            try:
                dt = pd.to_datetime(published_at, utc=True).to_pydatetime()
            except Exception:
                continue
            yield dt, title, body


def _iter_news_table(database_url: str, limit: int) -> Iterable[Tuple[datetime, str, str]]:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select published_at, title, summary
                from news_events
                order by published_at desc
                limit %s
                """,
                (int(limit),),
            )
            rows = cur.fetchall()
    for published_at, title, summary in rows:
        if published_at is None or not title:
            continue
        dt = pd.to_datetime(published_at, utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        yield dt.to_pydatetime(), str(title), str(summary or "")


def _find_event_anchor_index(market_df: pd.DataFrame, event_dt: datetime) -> Optional[int]:
    idx = market_df.index
    ts = pd.Timestamp(event_dt).tz_convert("UTC")
    pos = int(idx.searchsorted(ts, side="left"))
    if pos >= len(idx):
        return None
    return pos


def _forward_outcome(
    prices: pd.Series, times: pd.DatetimeIndex, anchor_pos: Optional[int], horizon: int
) -> Tuple[Optional[float], Optional[datetime]]:
    if anchor_pos is None or anchor_pos + horizon >= len(prices):
        return None, None
    p0, p1 = float(prices.iloc[anchor_pos]), float(prices.iloc[anchor_pos + horizon])
    if not (0.0 < p0 < float("inf") and 0.0 < p1 < float("inf")):
        return None, None
    return p1 / p0 - 1.0, times[anchor_pos + horizon].to_pydatetime()


def _build_events(
    market_df: pd.DataFrame,
    event_stream: Iterable[Tuple[datetime, str, str]],
    encoder: SentenceTransformer,
    max_events: int,
) -> List[HistoricalEvent]:
    events: List[HistoricalEvent] = []
    times = market_availability_times(market_df.index)
    prices = market_df["Gold"]
    for event_dt, title, body in event_stream:
        anchor_pos = _find_event_anchor_index(market_df, event_dt)
        t1_ret, t1_at = _forward_outcome(prices, times, anchor_pos, 1)
        t7_ret, t7_at = _forward_outcome(prices, times, anchor_pos, 7)

        text = (title + "\n" + body).strip()
        emb = encoder.encode([text], normalize_embeddings=True)[0].astype(float).tolist()

        event_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{event_dt.date().isoformat()}|{title}")
        events.append(
            HistoricalEvent(
                event_id=event_id,
                event_date=event_dt,
                headline=title,
                context_summary=body,
                embedding=emb,
                gold_t1_return=t1_ret,
                gold_t7_return=t7_ret,
                gold_t1_available_at=t1_at,
                gold_t7_available_at=t7_at,
            )
        )
        if len(events) >= max_events:
            break
    return events


def _ensure_schema(conn: psycopg.Connection, embedding_dim: int) -> str:
    with conn.cursor() as cur:
        cur.execute("select pg_advisory_xact_lock(hashtext('golden_sense_historical_events_schema'))")
        storage = _detect_storage(conn, allow_missing=True)
        if storage is None:
            storage = "array"
            cur.execute("select exists(select from pg_available_extensions where name = 'vector')")
            if cur.fetchone()[0]:
                try:
                    with conn.transaction():
                        cur.execute("create extension if not exists vector")
                    storage = "pgvector"
                except (psycopg.errors.InsufficientPrivilege, psycopg.errors.FeatureNotSupported):
                    # Roll back the extension attempt to its savepoint before
                    # creating the array table in the surrounding transaction.
                    pass

        if storage == "pgvector":
            cur.execute(
                f"""
                create table if not exists historical_events (
                    event_id uuid primary key,
                    event_date timestamptz not null,
                    headline text not null,
                    context_summary text not null,
                    embedding vector({embedding_dim}) not null,
                    gold_t1_return double precision not null,
                    gold_t7_return double precision not null
                )
                """
            )
            cur.execute(
                """
                create index if not exists historical_events_embedding_idx
                on historical_events using ivfflat (embedding vector_cosine_ops) with (lists = 50)
                """
            )
        else:
            cur.execute(
                """
                create table if not exists historical_events (
                    event_id uuid primary key,
                    event_date timestamptz not null,
                    headline text not null,
                    context_summary text not null,
                    embedding double precision[] not null,
                    gold_t1_return double precision not null,
                    gold_t7_return double precision not null
                )
                """
            )
        cur.execute(
            """
            alter table historical_events
                add column if not exists available_at timestamptz,
                add column if not exists gold_t1_available_at timestamptz,
                add column if not exists gold_t7_available_at timestamptz,
                alter column gold_t1_return drop not null,
                alter column gold_t7_return drop not null
            """
        )
        conn.commit()
        return storage


def _upsert_events(conn: psycopg.Connection, storage: str, events: Sequence[HistoricalEvent]) -> int:
    if not events:
        return 0
    captured_at = datetime.now(timezone.utc)
    embedding_sql = "%s::vector" if storage == "pgvector" else "%s"
    with conn.cursor() as cur:
        for event in events:
            embedding = (
                _vector_to_pgvector_literal(event.embedding) if storage == "pgvector" else event.embedding
            )
            cur.execute(
                f"""
                insert into historical_events (
                    event_id, event_date, headline, context_summary, embedding,
                    gold_t1_return, gold_t7_return, available_at,
                    gold_t1_available_at, gold_t7_available_at
                )
                values (%s, %s, %s, %s, {embedding_sql}, %s, %s, %s, %s, %s)
                on conflict (event_id) do update set
                    event_date = excluded.event_date,
                    headline = excluded.headline,
                    context_summary = excluded.context_summary,
                    embedding = excluded.embedding,
                    gold_t1_return = excluded.gold_t1_return,
                    gold_t7_return = excluded.gold_t7_return,
                    available_at = case
                        when historical_events.context_summary = excluded.context_summary
                         and historical_events.headline = excluded.headline
                         and historical_events.event_date = excluded.event_date
                        then coalesce(historical_events.available_at, excluded.available_at)
                        else excluded.available_at
                    end,
                    gold_t1_available_at = case
                        when historical_events.gold_t1_return = excluded.gold_t1_return
                        then coalesce(historical_events.gold_t1_available_at, excluded.gold_t1_available_at)
                        else greatest(excluded.gold_t1_available_at, excluded.available_at)
                    end,
                    gold_t7_available_at = case
                        when historical_events.gold_t7_return = excluded.gold_t7_return
                        then coalesce(historical_events.gold_t7_available_at, excluded.gold_t7_available_at)
                        else greatest(excluded.gold_t7_available_at, excluded.available_at)
                    end
                """,
                (
                    str(event.event_id), event.event_date, event.headline, event.context_summary, embedding,
                    event.gold_t1_return, event.gold_t7_return, captured_at,
                    event.gold_t1_available_at, event.gold_t7_available_at,
                ),
            )
    conn.commit()
    return len(events)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default="postgresql://localhost/postgres")
    parser.add_argument("--market-data", default="raw_market_data.csv")
    parser.add_argument("--events-jsonl", default="perception_layer/news_mock_data.jsonl")
    parser.add_argument("--events-source", choices=["jsonl", "news_table"], default="jsonl")
    parser.add_argument("--model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-events", type=int, default=80)
    args = parser.parse_args()

    market_path = Path(args.market_data)
    if not market_path.exists():
        raise SystemExit(f"market data not found: {market_path}")

    if args.events_source == "jsonl":
        news_path = Path(args.events_jsonl)
        if not news_path.exists():
            raise SystemExit(f"events jsonl not found: {news_path}")
        event_stream = _iter_news_jsonl(news_path)
    else:
        event_stream = _iter_news_table(args.database_url, limit=int(args.max_events))

    encoder = SentenceTransformer(args.model_id)
    embedding_dim = int(getattr(encoder, "get_sentence_embedding_dimension")())

    market_df = _load_market_data(market_path)
    t0 = perf_counter()
    events = _build_events(market_df, event_stream, encoder=encoder, max_events=int(args.max_events))
    t1 = perf_counter()

    if not events:
        raise SystemExit("no events built (check date overlap between market data and events)")

    with psycopg.connect(args.database_url) as conn:
        storage = _ensure_schema(conn, embedding_dim=embedding_dim)
        inserted = _upsert_events(conn, storage=storage, events=events)

    t2 = perf_counter()
    print(
        json.dumps(
            {
                "storage": storage,
                "embedding_dim": embedding_dim,
                "built_events": len(events),
                "inserted": inserted,
                "build_ms": int((t1 - t0) * 1000),
                "db_ms": int((t2 - t1) * 1000),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
