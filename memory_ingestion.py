from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import psycopg
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class HistoricalEvent:
    event_id: uuid.UUID
    event_date: datetime
    headline: str
    context_summary: str
    embedding: List[float]
    gold_t1_return: float
    gold_t7_return: float


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


def _compute_forward_returns(market_df: pd.DataFrame, anchor_pos: int) -> Optional[Tuple[float, float]]:
    if anchor_pos + 7 >= len(market_df):
        return None
    p0 = float(market_df["Gold"].iloc[anchor_pos])
    p1 = float(market_df["Gold"].iloc[anchor_pos + 1])
    p7 = float(market_df["Gold"].iloc[anchor_pos + 7])
    if p0 <= 0.0:
        return None
    return (p1 / p0 - 1.0), (p7 / p0 - 1.0)


def _build_events(
    market_df: pd.DataFrame,
    event_stream: Iterable[Tuple[datetime, str, str]],
    encoder: SentenceTransformer,
    max_events: int,
) -> List[HistoricalEvent]:
    events: List[HistoricalEvent] = []
    for event_dt, title, body in event_stream:
        anchor_pos = _find_event_anchor_index(market_df, event_dt)
        if anchor_pos is None:
            continue
        rets = _compute_forward_returns(market_df, anchor_pos)
        if rets is None:
            continue
        t1_ret, t7_ret = rets

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
                gold_t1_return=float(t1_ret),
                gold_t7_return=float(t7_ret),
            )
        )
        if len(events) >= max_events:
            break
    return events


def _ensure_schema(conn: psycopg.Connection, embedding_dim: int) -> str:
    with conn.cursor() as cur:
        cur.execute("select exists(select 1 from pg_available_extensions where name = 'vector')")
        vector_available = bool(cur.fetchone()[0])

        storage: str
        if vector_available:
            try:
                cur.execute("create extension if not exists vector")
                storage = "pgvector"
            except Exception:
                storage = "array"
        else:
            storage = "array"

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
        conn.commit()
        return storage


def _upsert_events(conn: psycopg.Connection, storage: str, events: Sequence[HistoricalEvent]) -> int:
    if not events:
        return 0
    with conn.cursor() as cur:
        if storage == "pgvector":
            for e in events:
                cur.execute(
                    """
                    insert into historical_events (
                        event_id, event_date, headline, context_summary, embedding, gold_t1_return, gold_t7_return
                    )
                    values (%s, %s, %s, %s, %s::vector, %s, %s)
                    on conflict (event_id) do update set
                        event_date = excluded.event_date,
                        headline = excluded.headline,
                        context_summary = excluded.context_summary,
                        embedding = excluded.embedding,
                        gold_t1_return = excluded.gold_t1_return,
                        gold_t7_return = excluded.gold_t7_return
                    """,
                    (
                        str(e.event_id),
                        e.event_date,
                        e.headline,
                        e.context_summary,
                        _vector_to_pgvector_literal(e.embedding),
                        e.gold_t1_return,
                        e.gold_t7_return,
                    ),
                )
        else:
            for e in events:
                cur.execute(
                    """
                    insert into historical_events (
                        event_id, event_date, headline, context_summary, embedding, gold_t1_return, gold_t7_return
                    )
                    values (%s, %s, %s, %s, %s, %s, %s)
                    on conflict (event_id) do update set
                        event_date = excluded.event_date,
                        headline = excluded.headline,
                        context_summary = excluded.context_summary,
                        embedding = excluded.embedding,
                        gold_t1_return = excluded.gold_t1_return,
                        gold_t7_return = excluded.gold_t7_return
                    """,
                    (
                        str(e.event_id),
                        e.event_date,
                        e.headline,
                        e.context_summary,
                        e.embedding,
                        e.gold_t1_return,
                        e.gold_t7_return,
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
