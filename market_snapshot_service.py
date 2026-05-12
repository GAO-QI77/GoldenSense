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
from fastapi.responses import JSONResponse

from data_loader import MarketDataProvider, create_market_data_provider
from service_contracts import (
    GoldPriceHistoryPoint,
    GoldPriceHistoryResponse,
    GoldPriceKeyNode,
    IndicatorCitation,
    IndicatorGroup,
    IndicatorItem,
    InstrumentSnapshot,
    MarketFeatureSummary,
    MarketIndicatorsResponse,
    MarketSnapshotResponse,
)


MARKET_SNAPSHOT_KEY = "golden_sense:market_snapshot"


@dataclass
class MarketSnapshotConfig:
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql://localhost/postgres"
    refresh_seconds: int = 60
    stale_after_seconds: int = 180
    allow_synthetic_fallback: bool = True
    provider_name: str = "yfinance"


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


def _moving_average(series: pd.Series, window: int) -> Optional[float]:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.tail(min(window, len(clean))).mean())


def _rsi(series: pd.Series, window: int = 14) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    delta = clean.diff().dropna()
    gain = delta.clip(lower=0).tail(window).mean()
    loss = -delta.clip(upper=0).tail(window).mean()
    if pd.isna(gain) or pd.isna(loss):
        return None
    if loss == 0:
        return 100.0
    rs = float(gain / loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    ema12 = clean.ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = clean.ewm(span=26, adjust=False).mean().iloc[-1]
    return float(ema12 - ema26)


def _atr_proxy_pct(series: pd.Series, window: int = 14) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    returns = clean.pct_change().abs().dropna()
    if returns.empty:
        return None
    return float(returns.tail(window).mean())


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
    status: str = "ok",
    degraded_reason: Optional[str] = None,
    source_freshness_seconds: Optional[int] = 0,
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
        ma5=_moving_average(gold_series, 5),
        ma20=_moving_average(gold_series, 20),
        ma60=_moving_average(gold_series, 60),
        rsi14=_rsi(gold_series),
        macd=_macd(gold_series),
        atr14_pct=_atr_proxy_pct(gold_series),
        stale_age_seconds=freshness_seconds,
        is_stale=False,
    )
    return MarketSnapshotResponse(
        asset="XAUUSD",
        as_of=as_of,
        freshness_seconds=freshness_seconds,
        stale_after_seconds=stale_after_seconds,
        is_stale=False,
        status=status,
        degraded_reason=degraded_reason,
        source_freshness_seconds=source_freshness_seconds,
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
    degraded_reason: str = "synthetic_fallback",
) -> MarketSnapshotResponse:
    return build_market_snapshot(
        build_synthetic_market_frame(now=now),
        source="synthetic_fallback",
        stale_after_seconds=stale_after_seconds,
        now=now,
        status="degraded",
        degraded_reason=degraded_reason,
        source_freshness_seconds=0,
    )


def _instrument(snapshot: MarketSnapshotResponse, symbol: str) -> Optional[InstrumentSnapshot]:
    return next((item for item in snapshot.instruments if item.symbol == symbol), None)


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:+.2f}%"


def _num(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _direction_from_value(value: Optional[float], *, positive_is_bullish: bool = True, threshold: float = 0.0) -> str:
    if value is None or abs(value) <= threshold:
        return "neutral"
    bullish = value > 0 if positive_is_bullish else value < 0
    return "bullish" if bullish else "bearish"


def _group_status(items: list[IndicatorItem], snapshot: MarketSnapshotResponse) -> str:
    if snapshot.status == "unavailable":
        return "unavailable"
    if snapshot.status != "ok" or snapshot.is_stale or any(item.status != "ok" for item in items):
        return "degraded"
    return "ok"


def _proxy_item(
    *,
    item_id: str,
    label: str,
    value: str,
    numeric_value: Optional[float],
    direction: str,
    source: str,
    source_url: str,
    freshness_seconds: int,
    degraded_reason: str = "proxy_static_source",
    unit: Optional[str] = None,
) -> IndicatorItem:
    return IndicatorItem(
        id=item_id,
        label=label,
        value=value,
        numeric_value=numeric_value,
        unit=unit,
        direction=direction,  # type: ignore[arg-type]
        source=source,
        source_url=source_url,
        freshness_seconds=freshness_seconds,
        status="degraded",
        degraded_reason=degraded_reason,
    )


def build_market_indicators(snapshot: MarketSnapshotResponse) -> MarketIndicatorsResponse:
    freshness = snapshot.freshness_seconds
    dxy = _instrument(snapshot, "DXY")
    vix = _instrument(snapshot, "VIX")
    us10y = _instrument(snapshot, "US10Y")
    us2y = _instrument(snapshot, "US2Y")
    spx = _instrument(snapshot, "SPX")
    wti = _instrument(snapshot, "WTI")
    momentum = snapshot.feature_summary.gold_momentum_5d
    divergence = snapshot.feature_summary.gold_usd_divergence
    yield_spread = snapshot.feature_summary.yield_curve_spread
    latest_price = snapshot.latest_price
    support = latest_price * 0.982
    resistance = latest_price * 1.018
    ma5 = snapshot.feature_summary.ma5
    ma20 = snapshot.feature_summary.ma20
    ma60 = snapshot.feature_summary.ma60
    rsi14 = snapshot.feature_summary.rsi14
    macd = snapshot.feature_summary.macd
    atr14_pct = snapshot.feature_summary.atr14_pct
    ma_direction = (
        "bullish"
        if ma5 is not None and ma20 is not None and ma60 is not None and latest_price > ma5 > ma20 > ma60
        else "bearish"
        if ma5 is not None and ma20 is not None and ma60 is not None and latest_price < ma5 < ma20 < ma60
        else "neutral"
    )
    rsi_direction = "risk" if rsi14 is not None and (rsi14 >= 70 or rsi14 <= 30) else _direction_from_value(
        None if rsi14 is None else rsi14 - 50,
        threshold=5.0,
    )

    fundamental_items = [
        _proxy_item(
            item_id="central-bank-demand",
            label="央行购金",
            value="244t Q1 proxy",
            numeric_value=244.0,
            unit="tonnes",
            direction="bullish",
            source="World Gold Council proxy",
            source_url="https://www.gold.org/goldhub/research/gold-demand-trends",
            freshness_seconds=86400,
        ),
        _proxy_item(
            item_id="gold-etf-holdings",
            label="黄金 ETF 持仓",
            value="monthly proxy",
            numeric_value=None,
            direction="neutral",
            source="World Gold Council proxy",
            source_url="https://www.gold.org/goldhub/data/gold-etfs-holdings-and-flows",
            freshness_seconds=86400,
        ),
        _proxy_item(
            item_id="investment-demand",
            label="投资需求",
            value="strong but proxy",
            numeric_value=None,
            direction="bullish",
            source="World Gold Council proxy",
            source_url="https://www.gold.org/goldhub/research/gold-demand-trends",
            freshness_seconds=86400,
        ),
        _proxy_item(
            item_id="supply-pressure",
            label="供应压力",
            value="not live",
            numeric_value=None,
            direction="neutral",
            source="WGC demand/supply proxy",
            source_url="https://www.gold.org/goldhub/research/gold-demand-trends",
            freshness_seconds=86400,
        ),
    ]

    technical_items = [
        IndicatorItem(
            id="technical-state",
            label="趋势状态",
            value=snapshot.feature_summary.technical_state,
            numeric_value=None,
            unit=None,
            direction={"bullish": "bullish", "bearish": "bearish"}.get(snapshot.feature_summary.technical_state, "neutral"),  # type: ignore[arg-type]
            source="market_snapshot_service",
            source_url=None,
            freshness_seconds=freshness,
            status=snapshot.status if snapshot.status != "unavailable" else "unavailable",
            degraded_reason=snapshot.degraded_reason,
        ),
        IndicatorItem(
            id="ma-5-20-60",
            label="MA5/20/60",
            value=f"{_num(ma5)} / {_num(ma20)} / {_num(ma60)}",
            numeric_value=None if ma20 is None else latest_price - ma20,
            unit="USD/oz",
            direction=ma_direction,  # type: ignore[arg-type]
            source="market_snapshot_service",
            source_url=None,
            freshness_seconds=freshness,
            status=snapshot.status if snapshot.status != "unavailable" else "unavailable",
            degraded_reason=snapshot.degraded_reason,
        ),
        IndicatorItem(
            id="rsi-14",
            label="RSI",
            value=_num(rsi14, 1),
            numeric_value=rsi14,
            unit="index",
            direction=rsi_direction,  # type: ignore[arg-type]
            source="market_snapshot_service",
            source_url=None,
            freshness_seconds=freshness,
            status=snapshot.status if snapshot.status != "unavailable" else "unavailable",
            degraded_reason=snapshot.degraded_reason,
        ),
        IndicatorItem(
            id="macd",
            label="MACD",
            value=_num(macd, 2),
            numeric_value=macd,
            unit="USD/oz",
            direction=_direction_from_value(macd),  # type: ignore[arg-type]
            source="market_snapshot_service",
            source_url=None,
            freshness_seconds=freshness,
            status=snapshot.status if snapshot.status != "unavailable" else "unavailable",
            degraded_reason=snapshot.degraded_reason,
        ),
        IndicatorItem(
            id="atr-volatility",
            label="ATR/波动率",
            value=_pct(atr14_pct),
            numeric_value=atr14_pct,
            unit="pct",
            direction="risk" if atr14_pct is not None and atr14_pct >= 0.02 else "neutral",
            source=vix.source if vix else "market_snapshot_service",
            source_url=None,
            freshness_seconds=freshness,
            status=snapshot.status if snapshot.status != "unavailable" else "unavailable",
            degraded_reason=snapshot.degraded_reason,
        ),
        IndicatorItem(
            id="support-resistance",
            label="关键支撑阻力",
            value=f"{support:.2f} / {resistance:.2f}",
            numeric_value=latest_price,
            unit="USD/oz",
            direction="neutral",
            source="market_snapshot_service proxy",
            source_url=None,
            freshness_seconds=freshness,
            status="degraded",
            degraded_reason="derived_support_resistance_proxy",
        ),
        IndicatorItem(
            id="gold-momentum-5d",
            label="5日动量",
            value=_pct(momentum),
            numeric_value=momentum,
            unit="pct",
            direction=_direction_from_value(momentum),  # type: ignore[arg-type]
            source="market_snapshot_service",
            source_url=None,
            freshness_seconds=freshness,
            status=snapshot.status if snapshot.status != "unavailable" else "unavailable",
            degraded_reason=snapshot.degraded_reason,
        ),
    ]

    macro_items = [
        IndicatorItem(
            id="dxy-change-1d",
            label="美元指数",
            value=f"{_num(dxy.price if dxy else None)} ({_pct(dxy.change_pct_1d if dxy else None)})",
            numeric_value=dxy.change_pct_1d if dxy else None,
            unit="pct",
            direction=_direction_from_value(dxy.change_pct_1d if dxy else None, positive_is_bullish=False),  # type: ignore[arg-type]
            source=dxy.source if dxy else "market_snapshot_service",
            source_url=None,
            freshness_seconds=freshness,
            status=snapshot.status if snapshot.status != "unavailable" else "unavailable",
            degraded_reason=snapshot.degraded_reason,
        ),
        IndicatorItem(
            id="yield-curve-spread",
            label="10Y-2Y 利差",
            value=f"{_num(yield_spread)} pct",
            numeric_value=yield_spread,
            unit="pct",
            direction="bullish" if yield_spread is not None and yield_spread < 0 else "neutral",
            source="market_snapshot_service",
            source_url=None,
            freshness_seconds=freshness,
            status=snapshot.status if snapshot.status != "unavailable" else "unavailable",
            degraded_reason=snapshot.degraded_reason,
        ),
        _proxy_item(
            item_id="real-yield-10y",
            label="10Y 实际利率",
            value=f"{_num((us10y.price - 2.3) if us10y else None)} pct proxy",
            numeric_value=(us10y.price - 2.3) if us10y else None,
            unit="pct",
            direction="bearish" if us10y and us10y.price > 4.0 else "neutral",
            source="FRED/TIPS proxy",
            source_url="https://fred.stlouisfed.org/series/DFII10",
            freshness_seconds=freshness,
            degraded_reason="real_yield_proxy_from_nominal_yield",
        ),
        _proxy_item(
            item_id="fedwatch-next-meeting",
            label="FedWatch",
            value="policy probability proxy",
            numeric_value=None,
            direction="neutral",
            source="CME FedWatch proxy",
            source_url="https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
            freshness_seconds=86400,
        ),
    ]

    flow_items = [
        _proxy_item(
            item_id="cftc-managed-money",
            label="CFTC Managed Money",
            value="weekly proxy",
            numeric_value=None,
            direction="neutral",
            source="CFTC COT proxy",
            source_url="https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
            freshness_seconds=86400,
        ),
        _proxy_item(
            item_id="comex-activity",
            label="COMEX 成交/持仓",
            value="not live",
            numeric_value=None,
            direction="neutral",
            source="exchange activity proxy",
            source_url=None,
            freshness_seconds=86400,
        ),
        IndicatorItem(
            id="risk-appetite",
            label="风险偏好",
            value=f"VIX {_num(vix.price if vix else None)} / SPX {_pct(spx.change_pct_1d if spx else None)}",
            numeric_value=vix.price if vix else None,
            unit=None,
            direction="risk" if vix and vix.price >= 30 else "neutral",
            source=vix.source if vix else "market_snapshot_service",
            source_url=None,
            freshness_seconds=freshness,
            status=snapshot.status if snapshot.status != "unavailable" else "unavailable",
            degraded_reason=snapshot.degraded_reason,
        ),
        IndicatorItem(
            id="gold-usd-divergence",
            label="黄金/美元背离",
            value=_pct(divergence),
            numeric_value=divergence,
            unit="pct",
            direction=_direction_from_value(divergence),  # type: ignore[arg-type]
            source="market_snapshot_service",
            source_url=None,
            freshness_seconds=freshness,
            status=snapshot.status if snapshot.status != "unavailable" else "unavailable",
            degraded_reason=snapshot.degraded_reason,
        ),
    ]
    if wti is not None:
        flow_items.append(
            IndicatorItem(
                id="oil-risk-input",
                label="原油风险输入",
                value=f"{_num(wti.price)} ({_pct(wti.change_pct_1d)})",
                numeric_value=wti.change_pct_1d,
                unit="pct",
                direction=_direction_from_value(wti.change_pct_1d),  # type: ignore[arg-type]
                source=wti.source,
                source_url=None,
                freshness_seconds=freshness,
                status=snapshot.status if snapshot.status != "unavailable" else "unavailable",
                degraded_reason=snapshot.degraded_reason,
            )
        )

    group_specs = [
        ("fundamental", "基本面", "央行、ETF、投资需求和供应压力的结构性背景。", fundamental_items),
        ("technical", "技术面", "趋势、动量、波动率和关键区间。", technical_items),
        ("macro_policy", "宏观政策", "美元、利率、实际利率和政策概率。", macro_items),
        ("flow_sentiment", "资金情绪", "期货资金、ETF、风险偏好和新闻代理。", flow_items),
    ]
    groups: list[IndicatorGroup] = []
    for group_id, title, summary, items in group_specs:
        bullish = len([item for item in items if item.direction == "bullish"])
        bearish = len([item for item in items if item.direction == "bearish"])
        risk = len([item for item in items if item.direction == "risk"])
        score = max(-1.0, min(1.0, (bullish - bearish - risk * 0.5) / max(1, len(items))))
        status = _group_status(items, snapshot)
        groups.append(
            IndicatorGroup(
                id=group_id,  # type: ignore[arg-type]
                title=title,
                summary=summary,
                score=score,
                status=status,  # type: ignore[arg-type]
                freshness_seconds=freshness,
                degraded_reason=snapshot.degraded_reason if status != "ok" else None,
                indicators=items,
            )
        )

    response_status = "degraded" if snapshot.status != "ok" or snapshot.is_stale or any(group.status != "ok" for group in groups) else "ok"
    return MarketIndicatorsResponse(
        asset=snapshot.asset,
        as_of=snapshot.as_of,
        freshness_seconds=freshness,
        stale_after_seconds=snapshot.stale_after_seconds,
        status=response_status,  # type: ignore[arg-type]
        degraded_reason=snapshot.degraded_reason if response_status != "ok" else None,
        groups=groups,
        citations=[
            IndicatorCitation(
                id="ind-wgc",
                label="World Gold Council demand and ETF proxy",
                source_type="market_indicators",
                excerpt="基本面指标第一期以公开 WGC 数据方向建模，未配置实时付费源时明确标记为 proxy/degraded。",
                url="https://www.gold.org/goldhub/data/gold-etfs-holdings-and-flows",
            ),
            IndicatorCitation(
                id="ind-cftc",
                label="CFTC COT proxy",
                source_type="market_indicators",
                excerpt="资金情绪保留 CFTC managed money 占位契约，真实接入前不作为实时数据展示。",
                url="https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
            ),
            IndicatorCitation(
                id="ind-cme",
                label="CME FedWatch proxy",
                source_type="market_indicators",
                excerpt="政策概率保留 FedWatch 契约，真实接入前只作为宏观政策指标占位。",
                url="https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
            ),
            IndicatorCitation(
                id="ind-snapshot",
                label="Market snapshot derived indicators",
                source_type="market_snapshot",
                excerpt="技术面、美元、利率曲线、风险偏好来自当前市场快照或其派生值。",
                url=None,
            ),
        ],
    )


def _factor_change(market_df: pd.DataFrame, column: str, index_pos: int) -> Optional[float]:
    if column not in market_df.columns or index_pos <= 0:
        return None
    series = market_df[column].dropna()
    if len(series) <= index_pos:
        return None
    prev = _to_float(series.iloc[index_pos - 1])
    current = _to_float(series.iloc[index_pos])
    if prev in (None, 0) or current is None:
        return None
    return current / prev - 1.0


def _factor_label(name: str, change: Optional[float], *, positive_label: str, negative_label: str, threshold: float) -> Optional[str]:
    if change is None or abs(change) < threshold:
        return None
    return f"{positive_label if change > 0 else negative_label}（{name} {change * 100:+.2f}%）"


def _gold_node_factors(market_df: pd.DataFrame, index_pos: int, gold_change: float) -> list[str]:
    factors = [
        _factor_label("DXY", _factor_change(market_df, "USD_Index", index_pos), positive_label="美元走强压制黄金", negative_label="美元走弱支撑黄金", threshold=0.003),
        _factor_label("10Y", _factor_change(market_df, "10Y_Bond", index_pos), positive_label="美债收益率上行", negative_label="美债收益率回落", threshold=0.003),
        _factor_label("VIX", _factor_change(market_df, "VIX", index_pos), positive_label="避险波动升温", negative_label="风险波动回落", threshold=0.03),
        _factor_label("SPX", _factor_change(market_df, "S&P500", index_pos), positive_label="风险资产反弹", negative_label="风险资产承压", threshold=0.006),
        _factor_label("WTI", _factor_change(market_df, "Crude_Oil", index_pos), positive_label="能源价格上行推升通胀预期", negative_label="能源价格回落", threshold=0.015),
    ]
    out = [factor for factor in factors if factor]
    if not out:
        out.append("技术面突破/跌破后的动量延续")
    if gold_change > 0 and not any("支撑黄金" in item or "避险" in item or "通胀" in item for item in out):
        out.append("黄金价格自身动量增强")
    if gold_change < 0 and not any("压制黄金" in item or "收益率上行" in item for item in out):
        out.append("黄金价格自身动量转弱")
    return out[:4]


def build_gold_price_history(
    market_df: pd.DataFrame,
    *,
    source: str,
    now: Optional[datetime] = None,
) -> GoldPriceHistoryResponse:
    if market_df.empty or "Gold" not in market_df.columns:
        raise ValueError("market_history_missing_gold")

    clean = market_df.dropna(subset=["Gold"]).copy()
    if len(clean) < 2:
        raise ValueError("market_history_insufficient_gold_points")

    points: list[GoldPriceHistoryPoint] = []
    key_nodes: list[GoldPriceKeyNode] = []
    gold = clean["Gold"].astype(float)
    changes = gold.pct_change()
    for index_pos, (idx, price) in enumerate(gold.items()):
        change = None if pd.isna(changes.loc[idx]) else float(changes.loc[idx])
        date_label = pd.Timestamp(idx).date().isoformat()
        points.append(GoldPriceHistoryPoint(date=date_label, price=float(price), change_pct=change))
        if change is not None and abs(change) >= 0.02:
            factors = _gold_node_factors(clean, index_pos, change)
            direction = "up" if change > 0 else "down"
            reason = (
                f"黄金单日上涨 {change * 100:+.2f}%，主要因素：{'；'.join(factors)}。"
                if direction == "up"
                else f"黄金单日下跌 {change * 100:+.2f}%，主要因素：{'；'.join(factors)}。"
            )
            key_nodes.append(
                GoldPriceKeyNode(
                    date=date_label,
                    price=float(price),
                    change_pct=change,
                    direction=direction,
                    reason=reason,
                    factors=factors,
                )
            )

    return GoldPriceHistoryResponse(
        asset="XAUUSD",
        as_of=now or datetime.now(timezone.utc),
        source=source,
        points=points,
        key_nodes=key_nodes,
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
    data["source_freshness_seconds"] = age
    if any(item.get("source") == "synthetic_fallback" for item in data.get("instruments", [])):
        data["status"] = "degraded"
        data["degraded_reason"] = data.get("degraded_reason") or "synthetic_fallback"
    data["feature_summary"]["stale_age_seconds"] = age
    data["feature_summary"]["is_stale"] = age > stale_after_seconds
    return MarketSnapshotResponse(**data)


async def _resolve_snapshot(
    loader: MarketDataProvider,
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
        reason = f"synthetic_fallback:{type(exc).__name__}:{exc}"
        return (
            build_synthetic_market_snapshot(stale_after_seconds=cfg.stale_after_seconds, degraded_reason=reason),
            reason,
        )


async def _refresh_loop(app: FastAPI) -> None:
    cfg: MarketSnapshotConfig = app.state.cfg
    loader: MarketDataProvider = app.state.market_loader
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
    market_loader: Optional[MarketDataProvider] = None,
    persistence: Optional[SnapshotPersistence] = None,
    config: Optional[MarketSnapshotConfig] = None,
    start_background_task: Optional[bool] = None,
) -> FastAPI:
    app_env = os.environ.get("APP_ENV", "development").lower()
    cfg = config or MarketSnapshotConfig(
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        database_url=os.environ.get("DATABASE_URL", "postgresql://localhost/postgres"),
        refresh_seconds=int(os.environ.get("MARKET_REFRESH_SECONDS", "60")),
        stale_after_seconds=int(os.environ.get("MARKET_STALE_AFTER_SECONDS", "180")),
        allow_synthetic_fallback=os.environ.get(
            "MARKET_ALLOW_SYNTHETIC_FALLBACK",
            "1" if app_env == "development" else "0",
        ) != "0",
        provider_name=os.environ.get(
            "MARKET_DATA_PROVIDER",
            "yfinance" if app_env == "development" else "external_required",
        ),
    )
    background_enabled = (
        os.environ.get("MARKET_START_BACKGROUND_TASK", "1") != "0"
        if start_background_task is None
        else start_background_task
    )

    loader = market_loader or create_market_data_provider(cfg.provider_name)
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
            "provider": getattr(loader, "provider_name", cfg.provider_name),
            "has_snapshot": latest_snapshot is not None,
            "last_error": getattr(app.state, "last_error", None),
        }

    @app.get("/health/live")
    async def health_live() -> Dict[str, str]:
        return {"status": "ok", "service": "market_snapshot"}

    @app.get("/health/ready")
    async def health_ready() -> JSONResponse:
        snapshot = getattr(app.state, "latest_snapshot", None) or persistence.load()
        errors = []
        if snapshot is None and not cfg.allow_synthetic_fallback:
            errors.append("market_snapshot_unavailable")
        status = "ok" if not errors else "unavailable"
        return JSONResponse(
            status_code=200 if not errors else 503,
            content={
                "status": status,
                "provider": getattr(loader, "provider_name", cfg.provider_name),
                "allow_synthetic_fallback": cfg.allow_synthetic_fallback,
                "has_snapshot": snapshot is not None,
                "last_error": getattr(app.state, "last_error", None),
                "errors": errors,
            },
        )

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
            app.state.last_error = "synthetic_fallback:cache_miss"
            snapshot = build_synthetic_market_snapshot(
                stale_after_seconds=cfg.stale_after_seconds,
                degraded_reason=app.state.last_error,
            )
            persistence.save(snapshot)
        if snapshot is None:
            raise HTTPException(status_code=503, detail="market_snapshot_unavailable")
        snapshot = _with_freshness(snapshot, stale_after_seconds=cfg.stale_after_seconds)
        app.state.latest_snapshot = snapshot
        return snapshot

    @app.get("/api/v1/market/indicators/current", response_model=MarketIndicatorsResponse)
    async def get_current_market_indicators() -> MarketIndicatorsResponse:
        snapshot = await get_latest_market_snapshot()
        return build_market_indicators(snapshot)

    @app.get("/api/v1/market/gold/history", response_model=GoldPriceHistoryResponse)
    async def get_gold_price_history() -> GoldPriceHistoryResponse:
        try:
            market_df = await asyncio.to_thread(loader.fetch_data, "1y", "1d")
            return build_gold_price_history(
                market_df,
                source=getattr(loader, "provider_name", cfg.provider_name),
            )
        except Exception as exc:
            if not cfg.allow_synthetic_fallback:
                raise HTTPException(status_code=503, detail=f"market_history_unavailable: {exc}") from exc
            market_df = build_synthetic_market_frame()
            return build_gold_price_history(
                market_df,
                source="synthetic_fallback",
            )

    return app


app = create_app()
