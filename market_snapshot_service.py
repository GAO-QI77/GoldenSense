from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import psycopg
import redis
from fastapi import FastAPI, HTTPException

from data_loader import MarketDataLoader
from service_contracts import InstrumentSnapshot, MarketFeatureSummary, MarketSnapshotResponse


MARKET_SNAPSHOT_KEY = "golden_sense:market_snapshot"


@dataclass
class MarketSnapshotConfig:
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql://localhost/postgres"
    refresh_seconds: int = 60
    stale_after_seconds: int = 180
    allow_synthetic_fallback: bool = True


class SnapshotPersistence:
    def __init__(self, redis_url: str, database_url: str):
        self._database_url = database_url
        self._redis_client: Optional[redis.Redis] = None
        try:
            self._redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        except Exception:
            self._redis_client = None

    def save(self, snapshot: MarketSnapshotResponse) -> None:
        payload = snapshot.model_dump(mode="json")
        if self._redis_client is not None:
            try:
                self._redis_client.set(MARKET_SNAPSHOT_KEY, json.dumps(payload, ensure_ascii=False))
            except Exception:
                pass
        self._save_db(payload)

    def load(self) -> Optional[MarketSnapshotResponse]:
        if self._redis_client is None:
            return None
        try:
            raw = self._redis_client.get(MARKET_SNAPSHOT_KEY)
        except Exception:
            return None
        if not raw:
            return None
        try:
            return MarketSnapshotResponse(**json.loads(raw))
        except Exception:
            return None

    def _save_db(self, payload: Dict[str, Any]) -> None:
        try:
            with psycopg.connect(self._database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        create table if not exists market_snapshots (
                            as_of timestamptz primary key,
                            asset text not null,
                            latest_price double precision not null,
                            payload jsonb not null
                        )
                        """
                    )
                    cur.execute(
                        """
                        insert into market_snapshots (as_of, asset, latest_price, payload)
                        values (%s, %s, %s, %s::jsonb)
                        on conflict (as_of) do update set
                            asset = excluded.asset,
                            latest_price = excluded.latest_price,
                            payload = excluded.payload
                        """,
                        (
                            payload["as_of"],
                            payload["asset"],
                            payload["latest_price"],
                            json.dumps(payload, ensure_ascii=False),
                        ),
                    )
                conn.commit()
        except Exception:
            return


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _series_change_pct(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    prev = float(clean.iloc[-2])
    last = float(clean.iloc[-1])
    if prev == 0:
        return None
    return last / prev - 1.0


def _technical_state(gold: pd.Series) -> str:
    clean = gold.dropna()
    if len(clean) < 20:
        return "mixed"
    last = float(clean.iloc[-1])
    ma5 = float(clean.tail(5).mean())
    ma20 = float(clean.tail(20).mean())
    if last > ma5 > ma20:
        return "bullish"
    if last < ma5 < ma20:
        return "bearish"
    return "mixed"


def _volatility_regime(vix_value: Optional[float]) -> str:
    if vix_value is None:
        return "elevated"
    if vix_value >= 30.0:
        return "stress"
    if vix_value >= 20.0:
        return "elevated"
    return "calm"


def build_market_snapshot(
    market_df: pd.DataFrame,
    *,
    source: str = "yfinance",
    stale_after_seconds: int = 180,
    now: Optional[datetime] = None,
) -> MarketSnapshotResponse:
    if market_df.empty or "Gold" not in market_df.columns:
        raise ValueError("market_data_missing_gold")

    as_of = now or datetime.now(timezone.utc)
    gold_series = market_df["Gold"].dropna()
    latest_price = float(gold_series.iloc[-1])
    price_change_pct_1d = _series_change_pct(gold_series)

    mapping = {
        "Gold": ("XAUUSD", "黄金"),
        "USD_Index": ("DXY", "美元指数"),
        "VIX": ("VIX", "波动率指数"),
        "10Y_Bond": ("US10Y", "10年美债收益率"),
        "2Y_Bond": ("US2Y", "2年美债收益率"),
        "S&P500": ("SPX", "标普500"),
        "Crude_Oil": ("WTI", "原油"),
    }

    instruments = []
    for col, (symbol, label) in mapping.items():
        if col not in market_df.columns:
            continue
        series = market_df[col].dropna()
        if series.empty:
            continue
        instruments.append(
            InstrumentSnapshot(
                symbol=symbol,
                label=label,
                price=float(series.iloc[-1]),
                change_pct_1d=_series_change_pct(series),
                source=source,
                as_of=as_of,
            )
        )

    vix_value = next((item.price for item in instruments if item.symbol == "VIX"), None)
    gold_momentum_5d = None
    gold_usd_divergence = None
    if len(gold_series.dropna()) >= 6:
        gold_prev = float(gold_series.iloc[-6])
        if gold_prev != 0:
            gold_momentum_5d = latest_price / gold_prev - 1.0

    usd_series = market_df["USD_Index"].dropna() if "USD_Index" in market_df.columns else pd.Series(dtype=float)
    if not usd_series.empty and len(usd_series) >= 6:
        usd_prev = float(usd_series.iloc[-6])
        usd_last = float(usd_series.iloc[-1])
        usd_change = usd_last / usd_prev - 1.0 if usd_prev != 0 else 0.0
        if gold_momentum_5d is not None:
            gold_usd_divergence = gold_momentum_5d - usd_change

    yield_curve_spread = None
    if "10Y_Bond" in market_df.columns and "2Y_Bond" in market_df.columns:
        y10 = _to_float(market_df["10Y_Bond"].dropna().iloc[-1])
        y2 = _to_float(market_df["2Y_Bond"].dropna().iloc[-1])
        if y10 is not None and y2 is not None:
            yield_curve_spread = y10 - y2

    freshness_seconds = 0
    feature_summary = MarketFeatureSummary(
        technical_state=_technical_state(gold_series),
        volatility_regime=_volatility_regime(vix_value),
        yield_curve_spread=yield_curve_spread,
        gold_usd_divergence=gold_usd_divergence,
        gold_momentum_5d=gold_momentum_5d,
        stale_age_seconds=freshness_seconds,
        is_stale=False,
    )
    return MarketSnapshotResponse(
        asset="XAUUSD",
        as_of=as_of,
        freshness_seconds=freshness_seconds,
        stale_after_seconds=stale_after_seconds,
        is_stale=False,
        latest_price=latest_price,
        price_change_pct_1d=price_change_pct_1d,
        instruments=instruments,
        feature_summary=feature_summary,
    )


def build_synthetic_market_frame(*, now: Optional[datetime] = None) -> pd.DataFrame:
    as_of = now or datetime.now(timezone.utc)
    idx = pd.date_range(end=as_of, periods=40, freq="D")
    steps = pd.Series(range(len(idx)), index=idx, dtype=float)
    return pd.DataFrame(
        {
            "Gold": 2288.0 + steps * 1.9,
            "USD_Index": 105.2 - steps * 0.04,
            "VIX": 17.0 + steps * 0.03,
            "10Y_Bond": 4.45 - steps * 0.004,
            "2Y_Bond": 4.85 - steps * 0.0045,
            "S&P500": 5040.0 + steps * 2.1,
            "Crude_Oil": 78.0 + steps * 0.11,
        },
        index=idx,
    )


def build_synthetic_market_snapshot(
    *,
    stale_after_seconds: int,
    now: Optional[datetime] = None,
) -> MarketSnapshotResponse:
    return build_market_snapshot(
        build_synthetic_market_frame(now=now),
        source="synthetic_fallback",
        stale_after_seconds=stale_after_seconds,
        now=now,
    )


def _with_freshness(
    snapshot: MarketSnapshotResponse,
    *,
    stale_after_seconds: int,
    now: Optional[datetime] = None,
) -> MarketSnapshotResponse:
    current_time = now or datetime.now(timezone.utc)
    age = max(0, int((current_time - snapshot.as_of).total_seconds()))
    data = snapshot.model_dump()
    data["freshness_seconds"] = age
    data["stale_after_seconds"] = stale_after_seconds
    data["is_stale"] = age > stale_after_seconds
    data["feature_summary"]["stale_age_seconds"] = age
    data["feature_summary"]["is_stale"] = age > stale_after_seconds
    return MarketSnapshotResponse(**data)


async def _resolve_snapshot(
    loader: MarketDataLoader,
    cfg: MarketSnapshotConfig,
) -> tuple[MarketSnapshotResponse, Optional[str]]:
    try:
        market_df = await asyncio.to_thread(loader.fetch_data, "6mo", "1d")
        snapshot = build_market_snapshot(
            market_df,
            stale_after_seconds=cfg.stale_after_seconds,
        )
        return snapshot, None
    except Exception as exc:
        if not cfg.allow_synthetic_fallback:
            raise
        return (
            build_synthetic_market_snapshot(stale_after_seconds=cfg.stale_after_seconds),
            f"synthetic_fallback:{type(exc).__name__}:{exc}",
        )


async def _refresh_loop(app: FastAPI) -> None:
    cfg: MarketSnapshotConfig = app.state.cfg
    loader: MarketDataLoader = app.state.market_loader
    persistence: SnapshotPersistence = app.state.persistence
    while True:
        try:
            snapshot, fallback_error = await _resolve_snapshot(loader, cfg)
            persistence.save(snapshot)
            app.state.latest_snapshot = snapshot
            app.state.last_error = fallback_error
        except Exception as exc:
            app.state.last_error = str(exc)
        await asyncio.sleep(cfg.refresh_seconds)


def create_app(
    *,
    market_loader: Optional[MarketDataLoader] = None,
    persistence: Optional[SnapshotPersistence] = None,
    config: Optional[MarketSnapshotConfig] = None,
    start_background_task: Optional[bool] = None,
) -> FastAPI:
    cfg = config or MarketSnapshotConfig(
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        database_url=os.environ.get("DATABASE_URL", "postgresql://localhost/postgres"),
        refresh_seconds=int(os.environ.get("MARKET_REFRESH_SECONDS", "60")),
        stale_after_seconds=int(os.environ.get("MARKET_STALE_AFTER_SECONDS", "180")),
        allow_synthetic_fallback=os.environ.get("MARKET_ALLOW_SYNTHETIC_FALLBACK", "1") != "0",
    )
    background_enabled = (
        os.environ.get("MARKET_START_BACKGROUND_TASK", "1") != "0"
        if start_background_task is None
        else start_background_task
    )

    loader = market_loader or MarketDataLoader()
    persistence = persistence or SnapshotPersistence(cfg.redis_url, cfg.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.market_loader = loader
        app.state.persistence = persistence
        app.state.latest_snapshot = persistence.load()
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

    app = FastAPI(title="GoldenSense Market Snapshot Service", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        latest_snapshot = getattr(app.state, "latest_snapshot", None)
        return {
            "status": "ok",
            "has_snapshot": latest_snapshot is not None,
            "last_error": getattr(app.state, "last_error", None),
        }

    @app.post("/api/v1/market/snapshot/refresh", response_model=MarketSnapshotResponse)
    async def refresh_market_snapshot() -> MarketSnapshotResponse:
        try:
            snapshot, fallback_error = await _resolve_snapshot(loader, cfg)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"market_refresh_failed: {exc}") from exc
        persistence.save(snapshot)
        app.state.latest_snapshot = snapshot
        app.state.last_error = fallback_error
        return _with_freshness(snapshot, stale_after_seconds=cfg.stale_after_seconds)

    @app.get("/api/v1/market/snapshot/latest", response_model=MarketSnapshotResponse)
    async def get_latest_market_snapshot() -> MarketSnapshotResponse:
        snapshot = getattr(app.state, "latest_snapshot", None) or persistence.load()
        if snapshot is None and cfg.allow_synthetic_fallback:
            snapshot = build_synthetic_market_snapshot(stale_after_seconds=cfg.stale_after_seconds)
            app.state.last_error = "synthetic_fallback:cache_miss"
            persistence.save(snapshot)
        if snapshot is None:
            raise HTTPException(status_code=503, detail="market_snapshot_unavailable")
        snapshot = _with_freshness(snapshot, stale_after_seconds=cfg.stale_after_seconds)
        app.state.latest_snapshot = snapshot
        return snapshot

    return app


app = create_app()
