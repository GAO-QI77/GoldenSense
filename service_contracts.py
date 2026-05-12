from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class InstrumentSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    label: str
    price: float
    change_pct_1d: Optional[float]
    source: str
    as_of: datetime


class MarketFeatureSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    technical_state: Literal["bullish", "bearish", "mixed"]
    volatility_regime: Literal["calm", "elevated", "stress"]
    yield_curve_spread: Optional[float]
    gold_usd_divergence: Optional[float]
    gold_momentum_5d: Optional[float]
    ma5: Optional[float] = None
    ma20: Optional[float] = None
    ma60: Optional[float] = None
    rsi14: Optional[float] = None
    macd: Optional[float] = None
    atr14_pct: Optional[float] = None
    stale_age_seconds: int = Field(ge=0)
    is_stale: bool


class MarketSnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: Literal["XAUUSD"]
    as_of: datetime
    freshness_seconds: int = Field(ge=0)
    stale_after_seconds: int = Field(ge=1)
    is_stale: bool
    status: Literal["ok", "degraded", "unavailable"] = "ok"
    degraded_reason: Optional[str] = None
    source_freshness_seconds: Optional[int] = Field(default=None, ge=0)
    latest_price: float
    price_change_pct_1d: Optional[float]
    instruments: List[InstrumentSnapshot]
    feature_summary: MarketFeatureSummary


IndicatorGroupId = Literal["fundamental", "technical", "macro_policy", "flow_sentiment"]
IndicatorStatus = Literal["ok", "degraded", "unavailable"]
IndicatorDirection = Literal["bullish", "bearish", "neutral", "risk"]


class GoldPriceHistoryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    price: float
    change_pct: Optional[float] = None


class GoldPriceKeyNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str
    price: float
    change_pct: float
    direction: Literal["up", "down"]
    reason: str
    factors: List[str] = Field(min_length=1)


class GoldPriceHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: Literal["XAUUSD"]
    as_of: datetime
    source: str
    points: List[GoldPriceHistoryPoint] = Field(min_length=2)
    key_nodes: List[GoldPriceKeyNode] = Field(default_factory=list)


class IndicatorCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    source_type: str
    excerpt: str
    url: Optional[str] = None


class IndicatorItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    value: str
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    direction: IndicatorDirection
    source: str
    source_url: Optional[str] = None
    freshness_seconds: int = Field(ge=0)
    status: IndicatorStatus = "ok"
    degraded_reason: Optional[str] = None


class IndicatorGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: IndicatorGroupId
    title: str
    summary: str
    score: float = Field(ge=-1.0, le=1.0)
    status: IndicatorStatus = "ok"
    freshness_seconds: int = Field(ge=0)
    degraded_reason: Optional[str] = None
    indicators: List[IndicatorItem] = Field(min_length=1)


class MarketIndicatorsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: Literal["XAUUSD"]
    as_of: datetime
    freshness_seconds: int = Field(ge=0)
    stale_after_seconds: int = Field(ge=1)
    status: IndicatorStatus = "ok"
    degraded_reason: Optional[str] = None
    groups: List[IndicatorGroup] = Field(min_length=4, max_length=4)
    citations: List[IndicatorCitation] = Field(default_factory=list)


class NewsEventItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    published_at: datetime
    title: str
    summary: str
    source: str
    normalized_event: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    importance: float = Field(ge=0.0)
    categories: List[str]
    url: Optional[str] = None


class RecentNewsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    freshness_seconds: int = Field(ge=0)
    status: Literal["ok", "degraded", "unavailable"] = "ok"
    degraded_reason: Optional[str] = None
    source_freshness_seconds: Optional[int] = Field(default=None, ge=0)
    items: List[NewsEventItem]
