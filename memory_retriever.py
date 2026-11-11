from __future__ import annotations

import json
from datetime import datetime, timezone
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import psycopg
from sentence_transformers import SentenceTransformer


def _vector_to_pgvector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def _detect_storage(conn: psycopg.Connection, *, allow_missing: bool = False) -> Optional[str]:
    row = conn.execute(
        """
        select t.typname
        from pg_attribute a
        join pg_type t on t.oid = a.atttypid
        where a.attrelid = to_regclass('historical_events')
          and a.attname = 'embedding' and not a.attisdropped
        """
    ).fetchone()
    if row is None:
        if allow_missing:
            return None
        raise RuntimeError("historical_events table not found; run memory_ingestion.py first")
    storage = {"vector": "pgvector", "_float8": "array"}.get(row[0])
    if storage is None:
        raise RuntimeError("unsupported historical_events embedding type")
    return storage


def _query_pgvector(
    conn: psycopg.Connection, query_vec: Sequence[float], top_k: int, decision_at: datetime
) -> List[Dict[str, object]]:
    qlit = _vector_to_pgvector_literal(query_vec)
    with conn.cursor() as cur:
        cur.execute(
            """
            with visible as materialized (
                select event_id, event_date, headline, context_summary, embedding,
                       gold_t1_return, gold_t7_return, gold_t1_available_at, gold_t7_available_at
                from historical_events
                where event_date <= %(cutoff)s and available_at <= %(cutoff)s
            )
            select
                event_id::text,
                event_date,
                headline,
                context_summary,
                case when gold_t1_available_at <= %(cutoff)s then gold_t1_return end,
                case when gold_t7_available_at <= %(cutoff)s then gold_t7_return end,
                (1 - (embedding <=> %(vector)s::vector)) as similarity
            from visible
            order by embedding <=> %(vector)s::vector, event_id
            limit %(limit)s
            """,
            {"vector": qlit, "limit": int(top_k), "cutoff": decision_at},
        )
        rows = cur.fetchall()
    return [
        {
            "event_id": r[0],
            "event_date": r[1].isoformat() if r[1] is not None else None,
            "headline": r[2],
            "context_summary": r[3],
            "gold_t1_return": float(r[4]) if r[4] is not None else None,
            "gold_t7_return": float(r[5]) if r[5] is not None else None,
            "similarity": float(r[6]),
        }
        for r in rows
    ]


def _query_array_cosine(
    conn: psycopg.Connection, query_vec: Sequence[float], top_k: int, decision_at: datetime
) -> List[Dict[str, object]]:
    q = np.asarray([float(x) for x in query_vec], dtype=np.float32)
    with conn.cursor() as cur:
        cur.execute(
            """
            select
                event_id::text,
                event_date,
                headline,
                context_summary,
                case when gold_t1_available_at <= %(cutoff)s then gold_t1_return end,
                case when gold_t7_available_at <= %(cutoff)s then gold_t7_return end,
                embedding
            from historical_events
            where event_date <= %(cutoff)s and available_at <= %(cutoff)s
            """,
            {"cutoff": decision_at},
        )
        rows = cur.fetchall()

    scored: List[Tuple[float, Tuple[object, ...]]] = []
    for r in rows:
        emb_raw = r[6]
        if emb_raw is None:
            continue
        emb = np.asarray([float(x) for x in emb_raw], dtype=np.float32)
        if emb.shape != q.shape:
            continue
        sim = float(np.dot(emb, q))
        scored.append((sim, r))

    scored.sort(key=lambda t: (-t[0], t[1][0]))
    top = scored[: int(top_k)]
    out: List[Dict[str, object]] = []
    for sim, r in top:
        out.append(
            {
                "event_id": str(r[0]),
                "event_date": r[1].isoformat() if r[1] is not None else None,
                "headline": str(r[2]),
                "context_summary": str(r[3]),
                "gold_t1_return": float(r[4]) if r[4] is not None else None,
                "gold_t7_return": float(r[5]) if r[5] is not None else None,
                "similarity": float(sim),
            }
        )
    return out


class MemoryRetriever:
    def __init__(self, database_url: str = "postgresql://localhost/postgres", model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.database_url = database_url
        self.model_id = model_id
        self.encoder = SentenceTransformer(model_id)

    def search(
        self, query_text: str, top_k: int = 3, *, decision_at: Optional[datetime] = None
    ) -> dict:
        decision_at = decision_at or datetime.now(timezone.utc)
        t0 = perf_counter()
        qvec = self.encoder.encode([query_text], normalize_embeddings=True)[0].astype(float).tolist()
        t1 = perf_counter()

        with psycopg.connect(self.database_url) as conn:
            storage = _detect_storage(conn)
            t2 = perf_counter()
            if storage == "pgvector":
                results = _query_pgvector(conn, qvec, top_k=top_k, decision_at=decision_at)
            else:
                results = _query_array_cosine(conn, qvec, top_k=top_k, decision_at=decision_at)
            t3 = perf_counter()

        return {
            "query": query_text,
            "top_k": top_k,
            "storage": storage,
            "status": "ok",
            "degraded_reason": None,
            "source_freshness_seconds": None,
            "timing_ms": {
                "embed": int((t1 - t0) * 1000),
                "connect_and_detect": int((t2 - t1) * 1000),
                "db_query": int((t3 - t2) * 1000),
                "total": int((t3 - t0) * 1000),
            },
            "results": results,
        }

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default="postgresql://localhost/postgres")
    parser.add_argument("--model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--text", default="")
    args = parser.parse_args()

    query_text = str(args.text).strip()
    if not query_text:
        raise SystemExit("--text is required")

    retriever = MemoryRetriever(database_url=args.database_url, model_id=args.model_id)
    out = retriever.search(query_text, top_k=int(args.top_k))

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
