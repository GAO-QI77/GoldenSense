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
    stale_age_seconds: int = Field(ge=0)
    is_stale: bool


class MarketSnapshotResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: Literal["XAUUSD"]
    as_of: datetime
    freshness_seconds: int = Field(ge=0)
    stale_after_seconds: int = Field(ge=1)
    is_stale: bool
    latest_price: float
    price_change_pct_1d: Optional[float]
    instruments: List[InstrumentSnapshot]
    feature_summary: MarketFeatureSummary


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
