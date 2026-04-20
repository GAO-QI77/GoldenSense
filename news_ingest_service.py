from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg
import redis
from fastapi import FastAPI, HTTPException, Query

from data_loader import NewsDataLoader
from service_contracts import NewsEventItem, RecentNewsResponse


RECENT_NEWS_KEY = "golden_sense:recent_news"


@dataclass
class NewsIngestConfig:
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql://localhost/postgres"
    refresh_seconds: int = 300
    refresh_timeout_seconds: float = 5.0
    stale_cache_grace_seconds: int = 1800
    allow_sample_fallback: bool = True


class NewsPersistence:
    def __init__(self, redis_url: str, database_url: str):
        self._database_url = database_url
        self._redis_client: Optional[redis.Redis] = None
        try:
            self._redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        except Exception:
            self._redis_client = None

    def save(self, response: RecentNewsResponse) -> None:
        payload = response.model_dump(mode="json")
        if self._redis_client is not None:
            try:
                self._redis_client.set(RECENT_NEWS_KEY, json.dumps(payload, ensure_ascii=False))
            except Exception:
                pass
        self._save_db(payload)

    def load(self) -> Optional[RecentNewsResponse]:
        if self._redis_client is None:
            return None
        try:
            raw = self._redis_client.get(RECENT_NEWS_KEY)
        except Exception:
            return None
        if not raw:
            return None
        try:
            return RecentNewsResponse(**json.loads(raw))
        except Exception:
            return None

    def _save_db(self, payload: Dict[str, Any]) -> None:
        try:
            with psycopg.connect(self._database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        create table if not exists news_events (
                            event_id text primary key,
                            published_at timestamptz not null,
                            title text not null,
                            summary text not null,
                            source text not null,
                            normalized_event text not null,
                            sentiment_score double precision not null,
                            importance double precision not null,
                            categories jsonb not null,
                            url text
                        )
                        """
                    )
                    for item in payload["items"]:
                        cur.execute(
                            """
                            insert into news_events (
                                event_id, published_at, title, summary, source, normalized_event,
                                sentiment_score, importance, categories, url
                            )
                            values (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                            on conflict (event_id) do update set
                                published_at = excluded.published_at,
                                title = excluded.title,
                                summary = excluded.summary,
                                source = excluded.source,
                                normalized_event = excluded.normalized_event,
                                sentiment_score = excluded.sentiment_score,
                                importance = excluded.importance,
                                categories = excluded.categories,
                                url = excluded.url
                            """,
                            (
                                item["event_id"],
                                item["published_at"],
                                item["title"],
                                item["summary"],
                                item["source"],
                                item["normalized_event"],
                                item["sentiment_score"],
                                item["importance"],
                                json.dumps(item["categories"], ensure_ascii=False),
                                item["url"],
                            ),
                        )
                conn.commit()
        except Exception:
            return


def _normalize_text(value: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", value or "")
    collapsed = re.sub(r"\s+", " ", no_tags).strip()
    return collapsed


def _parse_datetime(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _score_to_sentiment(score: float) -> float:
    if score == 0:
        return 0.0
    scaled = score / 4.0
    return _clamp(scaled, -1.0, 1.0)


def _categories_from_item(item: Dict[str, Any]) -> List[str]:
    categories: List[str] = []
    if abs(float(item.get("rates", 0.0))) > 0.3:
        categories.append("rates")
    if abs(float(item.get("inflation", 0.0))) > 0.3:
        categories.append("inflation")
    if abs(float(item.get("risk", 0.0))) > 0.3:
        categories.append("geopolitics")
    if abs(float(item.get("fx", 0.0))) > 0.3:
        categories.append("fx")
    if not categories:
        categories.append("macro")
    return categories


def normalize_news_items(news_items: List[Dict[str, Any]], *, now: Optional[datetime] = None) -> RecentNewsResponse:
    seen: set[str] = set()
    normalized: List[NewsEventItem] = []
    as_of = now or datetime.now(timezone.utc)
    for raw in news_items:
        title = _normalize_text(str(raw.get("title", "")))
        summary = _normalize_text(str(raw.get("summary", "")))
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        published = _parse_datetime(str(raw.get("published", "")))
        source = str(raw.get("source", "unknown"))
        normalized_event = title[:120]
        total_score = float(raw.get("total", 0.0))
        importance = max(0.1, min(1.0, abs(total_score) / 2.5 + (0.2 if "breaking" in title.lower() else 0.0)))
        digest = hashlib.sha1(f"{published.isoformat()}|{title}".encode("utf-8")).hexdigest()[:16]
        normalized.append(
            NewsEventItem(
                event_id=f"news-{digest}",
                published_at=published,
                title=title,
                summary=summary[:400],
                source=source,
                normalized_event=normalized_event,
                sentiment_score=_score_to_sentiment(total_score),
                importance=importance,
                categories=_categories_from_item(raw),
                url=raw.get("url") or raw.get("link"),
            )
        )
    normalized.sort(key=lambda item: (item.importance, item.published_at), reverse=True)
    return RecentNewsResponse(
        as_of=as_of,
        freshness_seconds=0,
        status="ok",
        degraded_reason=None,
        source_freshness_seconds=0,
        items=normalized,
    )


def build_sample_recent_news(*, now: Optional[datetime] = None) -> RecentNewsResponse:
    as_of = now or datetime.now(timezone.utc)
    items = [
        NewsEventItem(
            event_id="sample-news-rates",
            published_at=as_of,
            title="样本情景：市场等待美国通胀数据，金价维持高位震荡。",
            summary="这是本地回退新闻样本，用于在实时新闻源暂不可用时继续演示 Agent 的保守解释流程。",
            source="synthetic_fallback",
            normalized_event="美国通胀数据前的黄金震荡",
            sentiment_score=0.08,
            importance=0.45,
            categories=["inflation", "macro"],
            url=None,
        ),
        NewsEventItem(
            event_id="sample-news-fx",
            published_at=as_of,
            title="样本情景：美元短线走强，黄金上方空间受到压制。",
            summary="该条为回退情景，提醒系统在缺少实时新闻时仍保留美元与黄金反向关系的基本面语境。",
            source="synthetic_fallback",
            normalized_event="美元走强压制黄金",
            sentiment_score=-0.12,
            importance=0.4,
            categories=["fx"],
            url=None,
        ),
        NewsEventItem(
            event_id="sample-news-risk",
            published_at=as_of,
            title="样本情景：地缘风险未完全消退，避险需求仍有支撑。",
            summary="该条为回退情景，帮助前端和 Agent 在离线模式下仍能展示多空并存的解释结构。",
            source="synthetic_fallback",
            normalized_event="避险需求支撑黄金",
            sentiment_score=0.18,
            importance=0.5,
            categories=["geopolitics"],
            url=None,
        ),
    ]
    return RecentNewsResponse(
        as_of=as_of,
        freshness_seconds=0,
        status="degraded",
        degraded_reason="synthetic_fallback",
        source_freshness_seconds=0,
        items=items,
    )


def _mark_recent_news(
    payload: RecentNewsResponse,
    *,
    status: str,
    degraded_reason: Optional[str],
    now: Optional[datetime] = None,
) -> RecentNewsResponse:
    current_time = now or datetime.now(timezone.utc)
    age = max(0, int((current_time - payload.as_of).total_seconds()))
    data = payload.model_dump(mode="json")
    data["freshness_seconds"] = age
    data["status"] = status
    data["degraded_reason"] = degraded_reason
    data["source_freshness_seconds"] = age
    return RecentNewsResponse(**data)


def _with_freshness(payload: RecentNewsResponse, now: Optional[datetime] = None) -> RecentNewsResponse:
    status = payload.status
    degraded_reason = payload.degraded_reason
    if any(item.source == "synthetic_fallback" for item in payload.items[:1]):
        status = "degraded"
        degraded_reason = degraded_reason or "synthetic_fallback"
    return _mark_recent_news(payload, status=status, degraded_reason=degraded_reason, now=now)


def _is_cache_usable(payload: Optional[RecentNewsResponse], max_age_seconds: int) -> bool:
    if payload is None:
        return False
    return _with_freshness(payload).freshness_seconds <= max_age_seconds


async def _resolve_recent_news(
    loader: NewsDataLoader,
    cfg: NewsIngestConfig,
    cached: Optional[RecentNewsResponse] = None,
) -> tuple[RecentNewsResponse, Optional[str]]:
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(loader.fetch_news),
            timeout=cfg.refresh_timeout_seconds,
        )
        scored = await asyncio.to_thread(loader.analyze_causality, raw)
        recent = normalize_news_items(scored)
        if not recent.items:
            raise ValueError("news_feed_empty")
        return _mark_recent_news(recent, status="ok", degraded_reason=None), None
    except Exception as exc:
        if _is_cache_usable(cached, cfg.stale_cache_grace_seconds):
            reason = f"cache_fallback:{type(exc).__name__}:{exc}"
            return _mark_recent_news(cached, status="degraded", degraded_reason=reason), reason
        if not cfg.allow_sample_fallback:
            raise
        reason = f"synthetic_fallback:{type(exc).__name__}:{exc}"
        return _mark_recent_news(build_sample_recent_news(), status="degraded", degraded_reason=reason), reason


async def _refresh_loop(app: FastAPI) -> None:
    loader: NewsDataLoader = app.state.news_loader
    persistence: NewsPersistence = app.state.persistence
    refresh_seconds: int = app.state.cfg.refresh_seconds
    while True:
        try:
            cached = getattr(app.state, "latest_news", None) or persistence.load()
            recent, fallback_error = await _resolve_recent_news(loader, app.state.cfg, cached=cached)
            persistence.save(recent)
            app.state.latest_news = recent
            app.state.last_error = fallback_error
        except Exception as exc:
            app.state.last_error = str(exc)
        await asyncio.sleep(refresh_seconds)


def create_app(
    *,
    news_loader: Optional[NewsDataLoader] = None,
    persistence: Optional[NewsPersistence] = None,
    config: Optional[NewsIngestConfig] = None,
    start_background_task: Optional[bool] = None,
) -> FastAPI:
    cfg = config or NewsIngestConfig(
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        database_url=os.environ.get("DATABASE_URL", "postgresql://localhost/postgres"),
        refresh_seconds=int(os.environ.get("NEWS_REFRESH_SECONDS", "300")),
        refresh_timeout_seconds=float(os.environ.get("NEWS_FETCH_TIMEOUT_SECONDS", "5.0")),
        stale_cache_grace_seconds=int(os.environ.get("NEWS_STALE_CACHE_GRACE_SECONDS", "1800")),
        allow_sample_fallback=os.environ.get("NEWS_ALLOW_SAMPLE_FALLBACK", "1") != "0",
    )
    background_enabled = (
        os.environ.get("NEWS_START_BACKGROUND_TASK", "1") != "0"
        if start_background_task is None
        else start_background_task
    )
    loader = news_loader or NewsDataLoader()
    persistence = persistence or NewsPersistence(cfg.redis_url, cfg.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.news_loader = loader
        app.state.persistence = persistence
        app.state.latest_news = persistence.load()
        app.state.last_error = None
        task = None
        if background_enabled:
            task = asyncio.create_task(_refresh_loop(app))
        yield
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="GoldenSense News Ingest Service", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "has_news": getattr(app.state, "latest_news", None) is not None,
            "last_error": getattr(app.state, "last_error", None),
        }

    @app.post("/api/v1/news/refresh", response_model=RecentNewsResponse)
    async def refresh_recent_news() -> RecentNewsResponse:
        try:
            cached = getattr(app.state, "latest_news", None) or persistence.load()
            recent, fallback_error = await _resolve_recent_news(loader, cfg, cached=cached)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"news_refresh_failed: {exc}") from exc
        persistence.save(recent)
        app.state.latest_news = recent
        app.state.last_error = fallback_error
        return _with_freshness(recent)

    @app.get("/api/v1/news/recent", response_model=RecentNewsResponse)
    async def get_recent_news(
        limit: int = Query(default=8, ge=1, le=30),
        q: Optional[str] = Query(default=None, max_length=120),
    ) -> RecentNewsResponse:
        payload = getattr(app.state, "latest_news", None) or persistence.load()
        if payload is None and cfg.allow_sample_fallback:
            app.state.last_error = "synthetic_fallback:cache_miss"
            payload = _mark_recent_news(
                build_sample_recent_news(),
                status="degraded",
                degraded_reason=app.state.last_error,
            )
            persistence.save(payload)
        if payload is None:
            raise HTTPException(status_code=503, detail="recent_news_unavailable")
        query = (q or "").strip().lower()
        items = payload.items
        if query:
            filtered = [
                item
                for item in items
                if query in item.title.lower()
                or query in item.summary.lower()
                or query in item.normalized_event.lower()
            ]
            items = filtered or items
        data = payload.model_dump()
        data["items"] = [item.model_dump() for item in items[:limit]]
        return _with_freshness(RecentNewsResponse(**data))

    return app


app = create_app()
