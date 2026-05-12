from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict

import httpx
import psycopg
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from service_contracts import (
    GoldPriceHistoryPoint,
    GoldPriceHistoryResponse,
    GoldPriceKeyNode,
    IndicatorCitation,
    IndicatorGroup,
    IndicatorItem,
    MarketIndicatorsResponse,
    MarketSnapshotResponse,
    NewsEventItem,
    RecentNewsResponse,
)

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - optional dependency in local env
    AsyncOpenAI = None  # type: ignore[assignment]


class RiskResult(TypedDict):
    decision: Literal["PASS", "REJECTED", "EXECUTED", "EXEC_FAILED"]
    executed_position: float
    current_vix: Optional[float]
    vix_threshold: float
    notes: str


class AgentTriggerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    news_text: str = Field(min_length=1, max_length=5000)
    manual_vix: float = Field(ge=10.0, le=50.0)


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: Literal["T+1", "T+7"]
    reasoning_summary: str
    risk_warning: str


class RagEventItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str
    similarity: Optional[float]
    gold_t1_return: Optional[float]
    gold_t7_return: Optional[float]


class ImpactBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emotion_weight: float = Field(ge=-1.0, le=1.0)
    rag_consistency: float = Field(ge=0.0, le=1.0)
    quant_consistency: float = Field(ge=0.0, le=1.0)
    risk_reason: str = Field(max_length=200)


class AgentTriggerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: AgentDecision
    finbert_sentiment_score: float
    rag_top_3_event_titles: List[str]
    rag_top_3_events: List[RagEventItem]
    xgboost_probability: Optional[float]
    quant_probability: Optional[float]
    risk_result: RiskResult
    impact_breakdown: ImpactBreakdown
    timing_ms: Dict[str, int]


RiskProfile = Literal["conservative", "balanced", "aggressive"]
PublicHorizon = Literal["24h", "7d", "30d"]
SummaryStance = Literal["偏多", "偏空", "中性", "高风险观望"]
SummaryAction = Literal["观望", "小仓试探", "分批布局", "降低暴露"]
ConfidenceBand = Literal["低", "中", "高"]
ForecastBasis = Literal["ensemble_model", "heuristic_proxy", "degraded_fallback"]


class InvestorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_capacity: Literal["low", "medium", "high"]
    trading_horizon: Literal["short", "medium", "long"]
    experience_level: Literal["beginner", "intermediate", "advanced"]
    capital_allocation_pct: float = Field(ge=0.0, le=100.0)
    max_drawdown_pct: float = Field(ge=0.0, le=100.0)
    current_position: Literal["none", "long", "short", "hedged"]
    liquidity_need: Literal["low", "medium", "high"]
    leverage_attitude: Literal["none", "low", "medium", "high"]
    investment_goal: Literal["capital_preservation", "income", "event_trade", "trend_following", "speculation"]


class AgentAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=3000)
    optional_news_text: Optional[str] = Field(default=None, max_length=5000)
    risk_profile: RiskProfile = "conservative"
    horizon: PublicHorizon = "24h"
    locale: Literal["zh-CN"] = "zh-CN"
    investor_profile: Optional[InvestorProfile] = None


class SummaryCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stance: SummaryStance
    horizon: PublicHorizon
    confidence_band: ConfidenceBand
    action: SummaryAction
    reasons: List[str] = Field(min_length=2, max_length=4)
    invalidators: List[str] = Field(min_length=2, max_length=4)
    disclaimer: str


class EvidenceCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    signal_type: Literal["market", "quant", "news", "memory", "macro", "risk"]
    takeaway: str
    direction: Literal["supportive", "contradictory", "neutral"]
    citation_ids: List[str] = Field(min_length=1)


class CitationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    source_type: Literal["market_snapshot", "quant_forecast", "recent_news", "historical_analogs", "macro_context", "risk_profile"]
    excerpt: str
    url: Optional[str] = None


class RiskBanner(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["low", "medium", "high"]
    title: str
    message: str


class AgentAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    summary_card: SummaryCard
    horizon_forecasts: List["HorizonForecastCard"] = Field(min_length=3, max_length=3)
    recent_news: List[NewsEventItem] = Field(default_factory=list, max_length=6)
    evidence_cards: List[EvidenceCard] = Field(min_length=1)
    citations: List[CitationItem] = Field(min_length=1)
    risk_banner: RiskBanner
    degradation_flags: List[str] = Field(default_factory=list, max_length=8)
    follow_up_questions: List[str] = Field(min_length=1, max_length=4)
    timing_ms: Dict[str, int]


class AgentForecastsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    market_status: Dict[str, Any]
    horizon_forecasts: List["HorizonForecastCard"] = Field(min_length=3, max_length=3)
    degradation_flags: List[str] = Field(default_factory=list, max_length=8)
    timing_ms: Dict[str, int]


class DataSourceHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    source_type: str
    status: Literal["ok", "degraded", "unavailable"]
    freshness_seconds: int = Field(ge=0)
    expected_lag_seconds: int = Field(ge=1)
    cadence: str
    degraded_reason: Optional[str] = None
    coverage: List[str] = Field(min_length=1)
    url: Optional[str] = None


class AgentDashboardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    market_status: Dict[str, Any]
    horizon_forecasts: List["HorizonForecastCard"] = Field(min_length=3, max_length=3)
    indicator_groups: List[IndicatorGroup] = Field(min_length=4, max_length=4)
    gold_history: Optional[GoldPriceHistoryResponse] = None
    recent_news: List[NewsEventItem] = Field(default_factory=list, max_length=6)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    source_health: List[DataSourceHealth] = Field(default_factory=list)
    data_quality: Dict[str, Any]
    degradation_flags: List[str] = Field(default_factory=list, max_length=12)
    timing_ms: Dict[str, int]


class HorizonForecastCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon: PublicHorizon
    stance: SummaryStance
    confidence_band: ConfidenceBand
    action: SummaryAction
    probability: float = Field(ge=0.0, le=1.0)
    basis: ForecastBasis
    model_status: str = "unknown"
    model_loaded: bool = False
    model_checkpoint_path: Optional[str] = None
    reasons: List[str] = Field(min_length=2, max_length=4)


AgentAnalyzeResponse.model_rebuild()
AgentForecastsResponse.model_rebuild()
AgentDashboardResponse.model_rebuild()


class AgentFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str = Field(min_length=1, max_length=100)
    rating: Literal["helpful", "not_helpful"]
    comment: Optional[str] = Field(default=None, max_length=800)


class AgentFeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    status: Literal["recorded"]


class AgentTraceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_id: str
    request_payload: Dict[str, Any]
    tool_trace: List[Dict[str, Any]]
    evidence_payload: Dict[str, Any]
    response_payload: Dict[str, Any]
    feedback_rating: Optional[str]
    feedback_comment: Optional[str]
    created_at: str


class NarrativeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_card: SummaryCard
    risk_banner: RiskBanner
    follow_up_questions: List[str] = Field(min_length=1, max_length=4)


@dataclass
class AgentGatewayConfig:
    forecast_url: str
    memory_url: str
    market_snapshot_url: str
    market_indicators_url: str
    market_history_url: str
    recent_news_url: str
    default_model: str
    complex_model: str
    vix_circuit_breaker_threshold: float
    stale_after_seconds: int
    news_stale_after_seconds: int


@dataclass
class HistoricalEventsLookup:
    items: List[RagEventItem]
    status: str
    degraded_reason: Optional[str]
    source_freshness_seconds: Optional[int]


@dataclass
class AnalysisBundle:
    question: str
    optional_news_text: Optional[str]
    evidence_query: str
    horizon: PublicHorizon
    risk_profile: Dict[str, Any]
    investor_profile: Optional[Dict[str, Any]]
    risk_gate: Dict[str, Any]
    snapshot: MarketSnapshotResponse
    forecast: Dict[str, Any]
    news: RecentNewsResponse
    rag_events: List[RagEventItem]
    memory_status: str
    memory_degraded_reason: Optional[str]
    macro_context: Dict[str, Any]
    news_sentiment: float
    conflict_score: int
    has_conflict: bool
    quant_probability: Optional[float]
    xgboost_probability: Optional[float]
    quant_direction: int
    vix_value: Optional[float]
    is_high_risk: bool
    is_low_confidence: bool
    has_degraded_inputs: bool
    degradation_flags: List[str]
    tool_trace: List[Dict[str, Any]]


@dataclass
class AnalysisComputation:
    response: AgentAnalyzeResponse
    bundle: AnalysisBundle
    horizon_forecasts: List["HorizonForecastCard"]
    recent_news: List[NewsEventItem]
    evidence_cards: List[EvidenceCard]
    citations: List[CitationItem]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: Sequence[Optional[float]]) -> float:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def _confidence_to_score(band: ConfidenceBand) -> float:
    if band == "高":
        return 0.82
    if band == "中":
        return 0.61
    return 0.36


def _fallback_quant_forecast(reason: str) -> Dict[str, Any]:
    return {
        "direction_prediction": 0,
        "probability": 0.5,
        "xgboost_direction_prediction": 0,
        "xgboost_probability": 0.5,
        "service_status": "degraded",
        "forecast_basis": "degraded_fallback",
        "model_status": "unavailable",
        "model_loaded": False,
        "model_checkpoint_path": None,
        "supporting_reasons": ["量化服务暂不可用，系统已切换为保守中性占位。"],
        "reason": reason,
    }


def _fallback_recent_news(query: str) -> RecentNewsResponse:
    now = datetime.now(timezone.utc)
    prompt = query.strip() or "黄金市场"
    return RecentNewsResponse(
        as_of=now,
        freshness_seconds=0,
        status="degraded",
        degraded_reason="synthetic_fallback:recent_news_unavailable",
        source_freshness_seconds=0,
        items=[
            NewsEventItem(
                event_id="fallback-news-1",
                published_at=now,
                title=f"降级模式：围绕“{prompt[:32]}”的实时新闻暂不可用。",
                summary="系统已切换为保守解释模式，优先使用可用的市场快照、风险控制和历史逻辑框架。",
                source="synthetic_fallback",
                normalized_event=prompt[:80],
                sentiment_score=0.0,
                importance=0.1,
                categories=["fallback"],
                url=None,
            )
        ],
    )


def _fallback_market_indicators(snapshot: MarketSnapshotResponse, reason: str) -> MarketIndicatorsResponse:
    groups: List[IndicatorGroup] = []
    for group_id, title in [
        ("fundamental", "基本面"),
        ("technical", "技术面"),
        ("macro_policy", "宏观政策"),
        ("flow_sentiment", "资金情绪"),
    ]:
        groups.append(
            IndicatorGroup(
                id=group_id,  # type: ignore[arg-type]
                title=title,
                summary=f"{title}指标服务暂不可用，首页仅保留降级占位。",
                score=0.0,
                status="degraded",
                freshness_seconds=snapshot.freshness_seconds,
                degraded_reason=reason,
                indicators=[
                    IndicatorItem(
                        id=f"{group_id}-fallback-{idx}",
                        label=label,
                        value="unavailable",
                        numeric_value=None,
                        unit=None,
                        direction="neutral",
                        source="agent_gateway_fallback",
                        source_url=None,
                        freshness_seconds=snapshot.freshness_seconds,
                        status="degraded",
                        degraded_reason=reason,
                    )
                    for idx, label in enumerate(["数据源", "新鲜度", "方向", "风险"], start=1)
                ],
            )
        )
    return MarketIndicatorsResponse(
        asset=snapshot.asset,
        as_of=snapshot.as_of,
        freshness_seconds=snapshot.freshness_seconds,
        stale_after_seconds=snapshot.stale_after_seconds,
        status="degraded",
        degraded_reason=reason,
        groups=groups,
        citations=[
            IndicatorCitation(
                id="ind-fallback",
                label="Market indicators fallback",
                source_type="market_indicators",
                excerpt=f"指标服务不可用，Agent Gateway 已生成降级占位：{reason}",
                url=None,
            )
        ],
    )


def _fallback_gold_history(snapshot: MarketSnapshotResponse, reason: str) -> GoldPriceHistoryResponse:
    base = snapshot.latest_price
    as_of_date = snapshot.as_of.date()
    points = [
        GoldPriceHistoryPoint(date=as_of_date.isoformat(), price=base, change_pct=None),
        GoldPriceHistoryPoint(date=as_of_date.isoformat(), price=base, change_pct=0.0),
    ]
    return GoldPriceHistoryResponse(
        asset=snapshot.asset,
        as_of=snapshot.as_of,
        source=f"dashboard_fallback:{reason}",
        points=points,
        key_nodes=[],
    )


def _flatten_indicator_items(indicators: MarketIndicatorsResponse) -> List[IndicatorItem]:
    return [item for group in indicators.groups for item in group.indicators]


def _source_health_status(items: Sequence[IndicatorItem], fallback_status: str) -> str:
    if not items:
        return fallback_status if fallback_status in {"ok", "degraded", "unavailable"} else "unavailable"
    if any(item.status == "unavailable" for item in items):
        return "unavailable"
    if any(item.status != "ok" for item in items):
        return "degraded"
    return "ok"


def _source_health_freshness(items: Sequence[IndicatorItem], fallback_freshness: int) -> int:
    if not items:
        return max(0, fallback_freshness)
    return max(item.freshness_seconds for item in items)


def _source_health_reason(items: Sequence[IndicatorItem], fallback_reason: Optional[str]) -> Optional[str]:
    return next((item.degraded_reason for item in items if item.degraded_reason), fallback_reason)


def _build_dashboard_source_health(
    *,
    snapshot: MarketSnapshotResponse,
    indicators: MarketIndicatorsResponse,
    news: RecentNewsResponse,
    news_expected_lag_seconds: int,
) -> List[DataSourceHealth]:
    items = _flatten_indicator_items(indicators)
    wgc_items = [
        item
        for item in items
        if "World Gold Council" in item.source or item.source.startswith("WGC") or item.source_url and "gold.org" in item.source_url
    ]
    cftc_items = [item for item in items if "CFTC" in item.label or "CFTC" in item.source]
    cme_items = [item for item in items if "FedWatch" in item.label or "CME" in item.source]
    macro_market_items = [
        item
        for item in items
        if item.id in {"dxy-change-1d", "real-yield-10y", "yield-curve-spread"}
    ]

    market_status = snapshot.status
    market_reason = snapshot.degraded_reason
    if snapshot.is_stale and market_status == "ok":
        market_status = "degraded"
        market_reason = "market_snapshot_stale"

    return [
        DataSourceHealth(
            id="market_snapshot",
            label="Market Snapshot",
            source_type="market_snapshot",
            status=market_status,  # type: ignore[arg-type]
            freshness_seconds=snapshot.freshness_seconds,
            expected_lag_seconds=snapshot.stale_after_seconds,
            cadence="日内/分钟级",
            degraded_reason=market_reason,
            coverage=[item.symbol for item in snapshot.instruments[:6]] or [snapshot.asset],
            url=None,
        ),
        DataSourceHealth(
            id="wgc_gold_demand",
            label="WGC Gold Demand",
            source_type="fundamental",
            status=_source_health_status(wgc_items, indicators.status),  # type: ignore[arg-type]
            freshness_seconds=_source_health_freshness(wgc_items, indicators.freshness_seconds),
            expected_lag_seconds=2_678_400,
            cadence="月度/季度",
            degraded_reason=_source_health_reason(wgc_items, indicators.degraded_reason),
            coverage=["央行购金", "ETF flows", "实物/投资需求"],
            url="https://www.gold.org/goldhub/data/gold-etfs-holdings-and-flows",
        ),
        DataSourceHealth(
            id="cftc_cot",
            label="CFTC COT",
            source_type="flow_sentiment",
            status=_source_health_status(cftc_items, indicators.status),  # type: ignore[arg-type]
            freshness_seconds=_source_health_freshness(cftc_items, indicators.freshness_seconds),
            expected_lag_seconds=604_800,
            cadence="周度",
            degraded_reason=_source_health_reason(cftc_items, indicators.degraded_reason),
            coverage=["Managed Money", "投机净仓位", "拥挤度"],
            url="https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
        ),
        DataSourceHealth(
            id="cme_fedwatch",
            label="CME FedWatch",
            source_type="macro_policy",
            status=_source_health_status(cme_items, indicators.status),  # type: ignore[arg-type]
            freshness_seconds=_source_health_freshness(cme_items, indicators.freshness_seconds),
            expected_lag_seconds=86_400,
            cadence="日内",
            degraded_reason=_source_health_reason(cme_items, indicators.degraded_reason),
            coverage=["政策概率", "FOMC 路径", "降息预期"],
            url="https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
        ),
        DataSourceHealth(
            id="macro_market_rates",
            label="DXY / Real Yield",
            source_type="macro_policy",
            status=_source_health_status(macro_market_items, snapshot.status),  # type: ignore[arg-type]
            freshness_seconds=_source_health_freshness(macro_market_items, snapshot.freshness_seconds),
            expected_lag_seconds=snapshot.stale_after_seconds,
            cadence="日内",
            degraded_reason=_source_health_reason(macro_market_items, snapshot.degraded_reason),
            coverage=["DXY", "US10Y/US2Y", "10Y 实际利率代理"],
            url="https://fred.stlouisfed.org/series/DFII10",
        ),
        DataSourceHealth(
            id="recent_news",
            label="Recent News",
            source_type="news",
            status=news.status,  # type: ignore[arg-type]
            freshness_seconds=news.freshness_seconds,
            expected_lag_seconds=max(1, news_expected_lag_seconds),
            cadence="近实时",
            degraded_reason=news.degraded_reason,
            coverage=["宏观新闻", "政策语境", "地缘风险"],
            url=None,
        ),
    ]


def _forecast_is_degraded(payload: Dict[str, Any]) -> bool:
    if str(payload.get("service_status", "ok")) != "ok":
        return True
    return str(payload.get("model_status", "")) in {"loading", "unavailable"}


def _forecast_basis(payload: Dict[str, Any]) -> ForecastBasis:
    basis = str(payload.get("forecast_basis", "ensemble_model"))
    if basis in {"ensemble_model", "heuristic_proxy", "degraded_fallback"}:
        return basis  # type: ignore[return-value]
    return "degraded_fallback"


def _public_to_internal_horizon(horizon: PublicHorizon) -> str:
    if horizon == "24h":
        return "T+1"
    if horizon == "7d":
        return "T+7"
    return "T+30"


def _snapshot_uses_fallback(snapshot: MarketSnapshotResponse) -> bool:
    return snapshot.status != "ok" or any(item.source == "synthetic_fallback" for item in snapshot.instruments)


def _news_uses_fallback(news: RecentNewsResponse) -> bool:
    return any(item.source == "synthetic_fallback" for item in news.items[:1])


def _risk_profile_dict(profile: RiskProfile) -> Dict[str, Any]:
    mapping = {
        "conservative": {
            "profile": "conservative",
            "label": "保守型",
            "preferred_action": "小仓试探",
            "max_action": "小仓试探",
            "description": "优先保护本金，只接受分批、小仓位和明确止损框架。",
        },
        "balanced": {
            "profile": "balanced",
            "label": "平衡型",
            "preferred_action": "分批布局",
            "max_action": "分批布局",
            "description": "接受波动，但希望每一步都有证据和失效条件。",
        },
        "aggressive": {
            "profile": "aggressive",
            "label": "进取型",
            "preferred_action": "分批布局",
            "max_action": "分批布局",
            "description": "愿意承担更高波动，但仍需遵守节奏和风险边界。",
        },
    }
    return mapping[profile]


def _investor_profile_gate(profile: Optional[InvestorProfile], question: str = "") -> Dict[str, Any]:
    lowered_question = question.lower()
    prompt_safety_hits = [
        token
        for token in [
            "ignore previous",
            "ignore all",
            "system prompt",
            "隐藏风险",
            "忽略",
            "保证",
            "稳赚",
            "必赚",
            "无风险",
            "guarantee",
            "guaranteed",
            "risk-free",
            "hide risk",
        ]
        if token in lowered_question or token in question
    ]
    if profile is None:
        force_prompt_guard = bool(prompt_safety_hits)
        return {
            "investor_profile": None,
            "score": 5 if force_prompt_guard else 0,
            "level": "high" if force_prompt_guard else "low",
            "force_observation": force_prompt_guard,
            "notes": (
                ["安全护栏检测到提示注入或确定性收益语言，系统禁止输出激进建议。"]
                if force_prompt_guard
                else ["未填写完整问卷，系统仅使用基础风险偏好。"]
            ),
            "prompt_safety_hits": prompt_safety_hits,
        }

    payload = profile.model_dump()
    score = 0
    notes: List[str] = []
    allocation = profile.capital_allocation_pct
    drawdown = profile.max_drawdown_pct
    if allocation >= 50:
        score += 3
        notes.append("黄金计划资金占比过高。")
    elif allocation >= 25:
        score += 2
        notes.append("黄金计划资金占比偏高。")
    elif allocation >= 10:
        score += 1
        notes.append("黄金计划资金占比需要分步执行。")

    if drawdown <= 5:
        score += 2
        notes.append("最大可承受回撤很低。")
    elif drawdown <= 10:
        score += 1
        notes.append("最大可承受回撤偏低。")

    leverage_points = {"none": 0, "low": 1, "medium": 2, "high": 3}[profile.leverage_attitude]
    score += leverage_points
    if leverage_points:
        notes.append("问卷显示存在杠杆意愿。")
    if profile.experience_level == "beginner":
        score += 1
        notes.append("交易经验不足，需要降低执行强度。")
    if profile.liquidity_need == "high":
        score += 2
        notes.append("流动性需求较高，不适合激进暴露。")
    if profile.current_position in {"long", "short"}:
        score += 1
        notes.append("已有方向性持仓，需要先管理现有风险。")
    if profile.investment_goal == "speculation":
        score += 1
        notes.append("投资目标偏投机，需限制追涨杀跌。")
    if prompt_safety_hits:
        score += 5
        notes.append("安全护栏检测到提示注入或确定性收益语言。")

    level = "high" if score >= 5 else "medium" if score >= 2 else "low"
    return {
        "investor_profile": payload,
        "score": score,
        "level": level,
        "force_observation": score >= 5,
        "notes": notes or ["完整问卷未触发额外风险门控。"],
        "prompt_safety_hits": prompt_safety_hits,
    }


LOGGER = logging.getLogger("goldensense.agent_gateway")


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _evidence_query(req: AgentAnalyzeRequest) -> str:
    primary = (req.optional_news_text or "").strip()
    if primary:
        return primary
    return req.question.strip()


def _news_search_query(req: AgentAnalyzeRequest) -> str:
    primary = (req.optional_news_text or "").strip()
    if primary:
        return primary
    base = req.question.strip()
    gold_research_terms = "黄金 金价 美元 利率 美联储 ETF CFTC"
    return f"{base} {gold_research_terms}".strip()


def _memory_unavailable_copy(status: str, reason: Optional[str]) -> str:
    if status == "unavailable" and reason and ("not_started" in reason or "not_loaded" in reason):
        return "历史类比未启用：尚未初始化 historical_events 检索库，本轮不把它作为方向证据。"
    return f"历史记忆检索当前处于 {status} 状态，原因：{reason or '未返回'}。"


def _degradation_flags(
    *,
    snapshot: MarketSnapshotResponse,
    news: RecentNewsResponse,
    forecast: Dict[str, Any],
    memory_lookup: HistoricalEventsLookup,
) -> List[str]:
    flags: List[str] = []
    if snapshot.is_stale:
        flags.append("market_snapshot_stale")
    if _snapshot_uses_fallback(snapshot):
        flags.append("market_snapshot_synthetic_fallback")
    if _forecast_is_degraded(forecast):
        flags.append("quant_forecast_degraded")
    if news.status != "ok":
        flags.append(f"news_{news.status}")
    if _news_uses_fallback(news):
        flags.append("news_synthetic_fallback")
    if memory_lookup.status != "ok":
        flags.append(f"historical_memory_{memory_lookup.status}")
    return flags


def _forecast_degradation_flags(
    *,
    snapshot: MarketSnapshotResponse,
    forecast_map: Dict[PublicHorizon, Dict[str, Any]],
) -> List[str]:
    flags: List[str] = []
    if snapshot.is_stale:
        flags.append("market_snapshot_stale")
    if _snapshot_uses_fallback(snapshot):
        flags.append("market_snapshot_synthetic_fallback")
    if any(_forecast_is_degraded(item) for item in forecast_map.values()):
        flags.append("quant_forecast_degraded")
    return flags


def _normalize_memory_lookup(payload: Any) -> HistoricalEventsLookup:
    if isinstance(payload, HistoricalEventsLookup):
        return payload
    if isinstance(payload, list):
        items = [item for item in payload if isinstance(item, RagEventItem)]
        return HistoricalEventsLookup(
            items=items,
            status="ok",
            degraded_reason=None,
            source_freshness_seconds=None,
        )
    return HistoricalEventsLookup(
        items=[],
        status="degraded",
        degraded_reason="invalid_memory_lookup_payload",
        source_freshness_seconds=None,
    )


class TraceStoreUnavailableError(RuntimeError):
    pass


class ApiKeyAuthorizer:
    def __init__(self, *, public_keys: Sequence[str], internal_keys: Sequence[str]):
        self._public_keys = {key for key in public_keys if key}
        self._internal_keys = {key for key in internal_keys if key}

    def _fingerprint(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]

    def _client_host(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip() or "unknown"
        if request.client and request.client.host:
            return request.client.host
        return "unknown"

    def authorize(self, request: Request, *, internal_only: bool) -> Dict[str, str]:
        raw_key = request.headers.get("X-API-Key", "").strip()
        if not raw_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error_code": "auth_required",
                    "message": "X-API-Key is required for this endpoint.",
                },
            )

        role: Optional[str] = None
        if raw_key in self._internal_keys:
            role = "internal"
        elif raw_key in self._public_keys:
            role = "public"

        if role is None:
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "invalid_api_key",
                    "message": "Provided API key is not recognized.",
                },
            )
        if internal_only and role != "internal":
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "internal_api_key_required",
                    "message": "This endpoint requires an internal API key.",
                },
            )

        fingerprint = self._fingerprint(raw_key)
        return {
            "role": role,
            "fingerprint": fingerprint,
            "client_host": self._client_host(request),
            "client_id": f"{role}:{fingerprint}:{self._client_host(request)}",
        }


class SlidingWindowRateLimiter:
    def __init__(self, *, limit: int, window_seconds: int):
        self._limit = max(0, int(limit))
        self._window_seconds = max(1, int(window_seconds))
        self._buckets: Dict[str, Deque[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, client_id: str) -> None:
        if self._limit <= 0:
            return

        now = datetime.now(timezone.utc).timestamp()
        async with self._lock:
            bucket = self._buckets.setdefault(client_id, deque())
            while bucket and now - bucket[0] >= self._window_seconds:
                bucket.popleft()
            if len(bucket) >= self._limit:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error_code": "rate_limit_exceeded",
                        "message": f"Analyze rate limit exceeded. Try again later (limit={self._limit}/{self._window_seconds}s).",
                    },
                )
            bucket.append(now)


class AgentTraceStore:
    def __init__(
        self,
        database_url: str,
        *,
        allow_memory_fallback: bool,
        memory_ttl_seconds: int,
        memory_max_items: int,
    ):
        self._database_url = database_url
        self._allow_memory_fallback = allow_memory_fallback
        self._memory_ttl_seconds = max(60, int(memory_ttl_seconds))
        self._memory_max_items = max(1, int(memory_max_items))
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._db_ready = False

    async def startup(self) -> None:
        try:
            await asyncio.to_thread(self._ensure_schema_sync)
            self._db_ready = True
        except Exception as exc:
            self._db_ready = False
            LOGGER.warning(
                "agent_trace_schema_init_failed database_url=%s allow_memory_fallback=%s error=%s",
                self._database_url,
                self._allow_memory_fallback,
                f"{type(exc).__name__}:{exc}",
            )
            if not self._allow_memory_fallback:
                raise TraceStoreUnavailableError("trace_store_schema_unavailable") from exc

    def _prune_memory(self) -> None:
        if not self._allow_memory_fallback:
            self._memory.clear()
            return
        now_ts = datetime.now(timezone.utc).timestamp()
        expired = [
            analysis_id
            for analysis_id, row in self._memory.items()
            if now_ts - float(row.get("created_at_epoch", now_ts)) > self._memory_ttl_seconds
        ]
        for analysis_id in expired:
            self._memory.pop(analysis_id, None)
        while len(self._memory) > self._memory_max_items:
            oldest = next(iter(self._memory))
            self._memory.pop(oldest, None)

    def _store_memory(self, row: Dict[str, Any]) -> None:
        if not self._allow_memory_fallback:
            return
        self._prune_memory()
        self._memory[row["analysis_id"]] = row
        self._prune_memory()

    def _memory_row(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        self._prune_memory()
        row = self._memory.get(analysis_id)
        if row is None:
            return None
        payload = dict(row)
        payload.pop("created_at_epoch", None)
        return payload

    async def persist_analysis(
        self,
        *,
        analysis_id: str,
        request_payload: Dict[str, Any],
        tool_trace: List[Dict[str, Any]],
        evidence_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> str:
        now = datetime.now(timezone.utc)
        row = {
            "analysis_id": analysis_id,
            "request_payload": request_payload,
            "tool_trace": tool_trace,
            "evidence_payload": evidence_payload,
            "response_payload": response_payload,
            "feedback_rating": None,
            "feedback_comment": None,
            "created_at": now.isoformat(),
            "created_at_epoch": now.timestamp(),
        }
        self._store_memory(row)
        if not self._db_ready:
            if self._allow_memory_fallback:
                return analysis_id
            raise TraceStoreUnavailableError("trace_store_unavailable")

        try:
            await asyncio.to_thread(self._persist_analysis_sync, row)
            return analysis_id
        except Exception as exc:
            LOGGER.warning(
                "agent_trace_persist_failed analysis_id=%s allow_memory_fallback=%s error=%s",
                analysis_id,
                self._allow_memory_fallback,
                f"{type(exc).__name__}:{exc}",
            )
            if self._allow_memory_fallback:
                return analysis_id
            raise TraceStoreUnavailableError("trace_store_persist_failed") from exc

    async def save_feedback(self, analysis_id: str, rating: str, comment: Optional[str]) -> bool:
        row = self._memory.get(analysis_id)
        if row is not None:
            row["feedback_rating"] = rating
            row["feedback_comment"] = comment

        if self._db_ready:
            try:
                updated = await asyncio.to_thread(self._save_feedback_sync, analysis_id, rating, comment)
                return updated or row is not None
            except Exception as exc:
                LOGGER.warning(
                    "agent_trace_feedback_failed analysis_id=%s allow_memory_fallback=%s error=%s",
                    analysis_id,
                    self._allow_memory_fallback,
                    f"{type(exc).__name__}:{exc}",
                )
                if row is not None and self._allow_memory_fallback:
                    return True
                if not self._allow_memory_fallback:
                    raise TraceStoreUnavailableError("trace_store_feedback_failed") from exc
        return row is not None

    async def load_trace(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        row = self._memory_row(analysis_id)
        if row is not None:
            return row
        if not self._db_ready:
            if self._allow_memory_fallback:
                return None
            raise TraceStoreUnavailableError("trace_store_unavailable")
        try:
            return await asyncio.to_thread(self._load_trace_sync, analysis_id)
        except Exception as exc:
            LOGGER.warning(
                "agent_trace_load_failed analysis_id=%s allow_memory_fallback=%s error=%s",
                analysis_id,
                self._allow_memory_fallback,
                f"{type(exc).__name__}:{exc}",
            )
            if self._allow_memory_fallback:
                return None
            raise TraceStoreUnavailableError("trace_store_load_failed") from exc

    def health(self) -> Dict[str, Any]:
        self._prune_memory()
        return {
            "db_ready": self._db_ready,
            "allow_memory_fallback": self._allow_memory_fallback,
            "memory_items": len(self._memory),
        }

    def _persist_analysis_sync(self, row: Dict[str, Any]) -> None:
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into agent_analysis_traces (
                        analysis_id,
                        request_payload,
                        tool_trace,
                        evidence_payload,
                        response_payload,
                        feedback_rating,
                        feedback_comment
                    )
                    values (%s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s)
                    """,
                    (
                        row["analysis_id"],
                        json.dumps(row["request_payload"], ensure_ascii=False),
                        json.dumps(row["tool_trace"], ensure_ascii=False),
                        json.dumps(row["evidence_payload"], ensure_ascii=False),
                        json.dumps(row["response_payload"], ensure_ascii=False),
                        None,
                        None,
                    ),
                )
            conn.commit()

    def _save_feedback_sync(self, analysis_id: str, rating: str, comment: Optional[str]) -> bool:
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update agent_analysis_traces
                    set feedback_rating = %s,
                        feedback_comment = %s,
                        feedback_at = now()
                    where analysis_id = %s
                    """,
                    (rating, comment, analysis_id),
                )
                updated = cur.rowcount > 0
            conn.commit()
        return updated

    def _load_trace_sync(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select
                        analysis_id,
                        request_payload,
                        tool_trace,
                        evidence_payload,
                        response_payload,
                        feedback_rating,
                        feedback_comment,
                        created_at
                    from agent_analysis_traces
                    where analysis_id = %s
                    """,
                    (analysis_id,),
                )
                fetched = cur.fetchone()
        if fetched is None:
            return None
        return {
            "analysis_id": fetched[0],
            "request_payload": fetched[1],
            "tool_trace": fetched[2],
            "evidence_payload": fetched[3],
            "response_payload": fetched[4],
            "feedback_rating": fetched[5],
            "feedback_comment": fetched[6],
            "created_at": fetched[7].isoformat() if hasattr(fetched[7], "isoformat") else str(fetched[7]),
        }

    def _ensure_schema_sync(self) -> None:
        with psycopg.connect(self._database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    create table if not exists agent_analysis_traces (
                        analysis_id text primary key,
                        request_payload jsonb not null,
                        tool_trace jsonb not null,
                        evidence_payload jsonb not null,
                        response_payload jsonb not null,
                        feedback_rating text,
                        feedback_comment text,
                        created_at timestamptz not null default now(),
                        feedback_at timestamptz
                    )
                    """
                )
            conn.commit()


def compute_impact_breakdown(
    *,
    emotion_weight: float,
    rag_events: List[RagEventItem],
    decision: AgentDecision,
    xgboost_probability: Optional[float],
    risk_result: RiskResult,
) -> ImpactBreakdown:
    ew = float(emotion_weight)
    if math.isnan(ew) or math.isinf(ew):
        raise ValueError("emotion_weight_not_finite")
    ew = _clamp(ew, -1.0, 1.0)

    action_dir = 0
    if decision.action == "BUY":
        action_dir = 1
    elif decision.action == "SELL":
        action_dir = -1

    t1_vals: List[float] = []
    for event in rag_events[:3]:
        if event.gold_t1_return is not None:
            t1_vals.append(float(event.gold_t1_return))
    mean_t1 = float(sum(t1_vals) / len(t1_vals)) if t1_vals else None

    if action_dir == 0 or mean_t1 is None:
        rag_cons = 0.5
    else:
        rag_cons = _sigmoid(50.0 * float(action_dir) * float(mean_t1))
    rag_cons = _clamp(float(rag_cons), 0.0, 1.0)

    if xgboost_probability is None or action_dir == 0:
        quant_cons = 0.5
    else:
        p_up = _clamp(float(xgboost_probability), 0.0, 1.0)
        quant_cons = p_up if action_dir == 1 else (1.0 - p_up)
    quant_cons = _clamp(float(quant_cons), 0.0, 1.0)

    risk_reason = str(risk_result.get("notes", ""))
    if len(risk_reason) > 200:
        risk_reason = risk_reason[:200]

    return ImpactBreakdown(
        emotion_weight=ew,
        rag_consistency=rag_cons,
        quant_consistency=quant_cons,
        risk_reason=risk_reason,
    )


class BaseSentimentScorer:
    def score(self, text: str) -> float:
        raise NotImplementedError


class KeywordSentimentScorer(BaseSentimentScorer):
    def __init__(self) -> None:
        self._positive = [
            "rate cut",
            "降息",
            "避险",
            "conflict",
            "war",
            "safe haven",
            "央行购金",
            "central bank buying",
            "inflation",
            "通胀超预期",
            "weaker dollar",
            "美元走弱",
        ]
        self._negative = [
            "rate hike",
            "加息",
            "hawkish",
            "higher for longer",
            "美元走强",
            "strong dollar",
            "real yields rise",
            "收益率走高",
            "risk-on",
            "ceasefire",
            "停火",
        ]

    def score(self, text: str) -> float:
        lowered = text.lower()
        score = 0.0
        for token in self._positive:
            if token in lowered:
                score += 0.18
        for token in self._negative:
            if token in lowered:
                score -= 0.18
        return _clamp(score, -1.0, 1.0)


class HttpResearchToolbox:
    def __init__(self, http: httpx.AsyncClient, cfg: AgentGatewayConfig):
        self._http = http
        self._cfg = cfg

    async def get_market_snapshot(self) -> MarketSnapshotResponse:
        refresh_url = self._cfg.market_snapshot_url.replace("/latest", "/refresh")
        try:
            resp = await self._http.post(refresh_url, json={})
            resp.raise_for_status()
            return MarketSnapshotResponse(**resp.json())
        except Exception:
            resp = await self._http.get(self._cfg.market_snapshot_url)
            resp.raise_for_status()
            return MarketSnapshotResponse(**resp.json())

    async def get_market_indicators(self) -> MarketIndicatorsResponse:
        resp = await self._http.get(self._cfg.market_indicators_url)
        resp.raise_for_status()
        return MarketIndicatorsResponse(**resp.json())

    async def get_gold_history(self) -> GoldPriceHistoryResponse:
        resp = await self._http.get(self._cfg.market_history_url)
        resp.raise_for_status()
        return GoldPriceHistoryResponse(**resp.json())

    async def get_quant_forecast(self, horizon: PublicHorizon) -> Dict[str, Any]:
        mapped = _public_to_internal_horizon(horizon)
        payload = {
            "asset_symbol": "XAUUSD",
            "horizon": mapped,
            "current_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        resp = await self._http.post(self._cfg.forecast_url, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_quant_forecasts(self, horizons: List[PublicHorizon]) -> Dict[PublicHorizon, Dict[str, Any]]:
        batch_url = (
            self._cfg.forecast_url.replace("/api/v1/forecast", "/api/v1/forecast/batch")
            if "/api/v1/forecast" in self._cfg.forecast_url
            else f"{self._cfg.forecast_url.rstrip('/')}/batch"
        )
        mapped_horizons = [_public_to_internal_horizon(horizon) for horizon in horizons]
        payload = {
            "asset_symbol": "XAUUSD",
            "horizons": mapped_horizons,
            "current_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        resp = await self._http.post(batch_url, json=payload)
        if resp.status_code in {404, 405}:
            return {horizon: await self.get_quant_forecast(horizon) for horizon in horizons}
        resp.raise_for_status()
        data = resp.json()
        forecasts = data.get("forecasts", {})
        return {
            horizon: forecasts[_public_to_internal_horizon(horizon)]
            for horizon in horizons
            if _public_to_internal_horizon(horizon) in forecasts
        }

    async def search_recent_news(self, query: str, limit: int = 6) -> RecentNewsResponse:
        cached: Optional[RecentNewsResponse] = None
        try:
            resp = await self._http.get(self._cfg.recent_news_url, params={"limit": limit, "q": query})
            resp.raise_for_status()
            cached = RecentNewsResponse(**resp.json())
            if cached.items and cached.freshness_seconds <= self._cfg.news_stale_after_seconds:
                return cached
        except Exception:
            cached = None

        refresh_url = self._cfg.recent_news_url.replace("/recent", "/refresh")
        try:
            refresh = await self._http.post(refresh_url, json={})
            refresh.raise_for_status()
            resp = await self._http.get(self._cfg.recent_news_url, params={"limit": limit, "q": query})
            resp.raise_for_status()
            return RecentNewsResponse(**resp.json())
        except Exception:
            if cached is not None:
                return cached
            resp = await self._http.get(self._cfg.recent_news_url, params={"limit": limit, "q": query})
            resp.raise_for_status()
            return RecentNewsResponse(**resp.json())

    async def retrieve_historical_events(self, text: str, top_k: int = 3) -> HistoricalEventsLookup:
        resp = await self._http.post(self._cfg.memory_url, json={"current_event_text": text, "top_k": top_k})
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("results", [])
        out: List[RagEventItem] = []
        if isinstance(items, list):
            for item in items[:top_k]:
                if isinstance(item, dict):
                    out.append(
                        RagEventItem(
                            headline=str(item.get("headline", "")),
                            similarity=_safe_float(item.get("similarity")),
                            gold_t1_return=_safe_float(item.get("gold_t1_return")),
                            gold_t7_return=_safe_float(item.get("gold_t7_return")),
                            )
                        )
        return HistoricalEventsLookup(
            items=out,
            status=str(payload.get("status", "ok")),
            degraded_reason=payload.get("degraded_reason"),
            source_freshness_seconds=payload.get("source_freshness_seconds"),
        )

    def get_macro_context(self, snapshot: MarketSnapshotResponse, news: RecentNewsResponse) -> Dict[str, Any]:
        divergence = snapshot.feature_summary.gold_usd_divergence
        if divergence is None:
            dollar_message = "美元方向暂时不清晰，对金价的压制/支撑证据有限。"
            macro_signal = 0
        elif divergence > 0:
            dollar_message = "黄金强于美元代理，说明避险或配置需求仍在支撑金价。"
            macro_signal = 1
        else:
            dollar_message = "美元代理更强，说明黄金短线承压。"
            macro_signal = -1

        curve = snapshot.feature_summary.yield_curve_spread
        if curve is None:
            rate_message = "利率曲线数据不足，暂不把利差作为核心判断。"
        elif curve < 0:
            rate_message = "收益率曲线倒挂仍在，宏观对冲需求没有消失。"
        else:
            rate_message = "收益率曲线为正，避险驱动相对没有那么极端。"

        avg_news_sentiment = _mean([item.sentiment_score for item in news.items[:5]])
        if avg_news_sentiment > 0.15:
            news_message = "近期新闻基调偏多，更多是对黄金有利的避险/通胀叙事。"
        elif avg_news_sentiment < -0.15:
            news_message = "近期新闻基调偏空，更多是强美元或鹰派利率叙事。"
        else:
            news_message = "近期新闻偏中性，暂未形成一边倒的基本面推动。"

        return {
            "macro_signal": macro_signal,
            "avg_news_sentiment": avg_news_sentiment,
            "dollar_message": dollar_message,
            "rate_message": rate_message,
            "news_message": news_message,
        }

    def get_user_risk_profile(self, profile: RiskProfile) -> Dict[str, Any]:
        return _risk_profile_dict(profile)


class OpenAINarrator:
    def __init__(self, cfg: AgentGatewayConfig):
        self._cfg = cfg
        self._client = AsyncOpenAI() if AsyncOpenAI is not None and os.environ.get("OPENAI_API_KEY") else None

    async def narrate(self, bundle: AnalysisBundle, draft: NarrativeOutput) -> NarrativeOutput:
        if self._client is None:
            return draft

        model = self._cfg.complex_model if bundle.has_conflict or bundle.is_high_risk else self._cfg.default_model
        prompt_payload = {
            "question": bundle.question,
            "optional_news_text": bundle.optional_news_text,
            "risk_profile": bundle.risk_profile,
            "snapshot": bundle.snapshot.model_dump(mode="json"),
            "forecast": bundle.forecast,
            "recent_news": bundle.news.model_dump(mode="json"),
            "rag_events": [event.model_dump() for event in bundle.rag_events],
            "macro_context": bundle.macro_context,
            "risk_gate": bundle.risk_gate,
            "draft": draft.model_dump(),
        }
        schema = NarrativeOutput.model_json_schema()
        try:
            response = await self._client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "你是 GoldenSense 的中文教育型黄金投资助手。"
                                    "你只能基于给定证据做结构化总结，不得编造来源，不得给出确定性喊单。"
                                    "如果证据冲突、数据陈旧或风险过高，必须保持保守措辞。"
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": json.dumps(prompt_payload, ensure_ascii=False)}],
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "goldensense_agent_narrative",
                        "schema": schema,
                    }
                },
            )
            output_text = getattr(response, "output_text", "")
            if not output_text:
                return draft
            return NarrativeOutput(**json.loads(output_text))
        except Exception:
            return draft


class AgentAnalysisService:
    def __init__(
        self,
        *,
        toolbox: HttpResearchToolbox,
        narrator: OpenAINarrator,
        sentiment_scorer: BaseSentimentScorer,
        trace_store: AgentTraceStore,
        cfg: AgentGatewayConfig,
    ):
        self._toolbox = toolbox
        self._narrator = narrator
        self._sentiment_scorer = sentiment_scorer
        self._trace_store = trace_store
        self._cfg = cfg

    async def analyze(self, req: AgentAnalyzeRequest) -> AgentAnalyzeResponse:
        return (await self.analyze_internal(req)).response

    async def analyze_internal(self, req: AgentAnalyzeRequest) -> AnalysisComputation:
        t0 = datetime.now(timezone.utc)
        evidence_query = _evidence_query(req)
        news_query = _news_search_query(req)
        snapshot, forecast_map, news, memory_lookup, tool_trace = await self._gather(
            req,
            evidence_query=evidence_query,
            news_query=news_query,
        )
        rag_events = memory_lookup.items
        news_sentiment = self._derive_news_sentiment(req, news)
        risk_profile = self._toolbox.get_user_risk_profile(req.risk_profile)
        risk_gate = _investor_profile_gate(req.investor_profile, req.question)
        if req.investor_profile is not None:
            risk_profile = dict(risk_profile)
            risk_profile["investor_profile"] = risk_gate["investor_profile"]
            risk_profile["force_observation"] = risk_gate["force_observation"]
            risk_profile["gate_level"] = risk_gate["level"]
            risk_profile["description"] = (
                f"{risk_profile['description']} 完整问卷："
                f"资金占比 {req.investor_profile.capital_allocation_pct:.1f}%，"
                f"最大可承受回撤 {req.investor_profile.max_drawdown_pct:.1f}%，"
                f"杠杆态度 {req.investor_profile.leverage_attitude}。"
            )
        macro_context = self._toolbox.get_macro_context(snapshot, news)
        selected_forecast = forecast_map[req.horizon]
        selected_outlook = self._evaluate_horizon_outlook(
            horizon=req.horizon,
            forecast=selected_forecast,
            snapshot=snapshot,
            news=news,
            news_sentiment=news_sentiment,
            rag_events=rag_events,
            memory_lookup=memory_lookup,
            macro_context=macro_context,
            risk_profile=risk_profile,
        )
        horizon_forecasts = [
            self._build_stable_horizon_forecast_card(
                horizon=horizon,
                forecast=forecast_map[horizon],
                snapshot=snapshot,
            )
            for horizon in ("24h", "7d", "30d")
        ]
        degradation_flags = _degradation_flags(
            snapshot=snapshot,
            news=news,
            forecast=selected_forecast,
            memory_lookup=memory_lookup,
        )

        bundle = AnalysisBundle(
            question=req.question,
            optional_news_text=req.optional_news_text,
            evidence_query=evidence_query,
            horizon=req.horizon,
            risk_profile=risk_profile,
            investor_profile=risk_gate["investor_profile"],
            risk_gate=risk_gate,
            snapshot=snapshot,
            forecast=selected_forecast,
            news=news,
            rag_events=rag_events,
            memory_status=memory_lookup.status,
            memory_degraded_reason=memory_lookup.degraded_reason,
            macro_context=macro_context,
            news_sentiment=news_sentiment,
            conflict_score=selected_outlook["conflict_score"],
            has_conflict=selected_outlook["has_conflict"],
            quant_probability=selected_outlook["probability"],
            xgboost_probability=selected_outlook["xgboost_probability"],
            quant_direction=selected_outlook["quant_direction"],
            vix_value=selected_outlook["vix_value"],
            is_high_risk=selected_outlook["is_high_risk"],
            is_low_confidence=selected_outlook["is_low_confidence"],
            has_degraded_inputs=selected_outlook["has_degraded_inputs"],
            degradation_flags=degradation_flags,
            tool_trace=tool_trace,
        )

        citations = self._build_citations(bundle)
        evidence_cards = self._build_evidence_cards(bundle, citations)
        if not evidence_cards or not citations:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "insufficient_evidence",
                    "message": "No evidence cards available for analysis.",
                },
            )

        draft = self._build_draft_narrative(req, bundle)
        narrative = await self._narrator.narrate(bundle, draft)
        if not evidence_cards:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "empty_evidence_cards",
                    "message": "Agent response must include evidence cards.",
                },
            )

        elapsed_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
        analysis_id = str(uuid.uuid4())
        response_model = AgentAnalyzeResponse(
            analysis_id=analysis_id,
            summary_card=narrative.summary_card,
            horizon_forecasts=horizon_forecasts,
            recent_news=news.items[:6],
            evidence_cards=evidence_cards,
            citations=citations,
            risk_banner=narrative.risk_banner,
            degradation_flags=degradation_flags,
            follow_up_questions=narrative.follow_up_questions,
            timing_ms={"total": elapsed_ms},
        )
        try:
            await self._trace_store.persist_analysis(
                analysis_id=analysis_id,
                request_payload=req.model_dump(mode="json"),
                tool_trace=tool_trace,
                evidence_payload={
                    "evidence_query": evidence_query,
                    "degradation_flags": degradation_flags,
                    "bundle": {
                        "snapshot": bundle.snapshot.model_dump(mode="json"),
                        "forecast": bundle.forecast,
                        "news": bundle.news.model_dump(mode="json"),
                        "rag_events": [event.model_dump() for event in bundle.rag_events],
                        "memory_status": bundle.memory_status,
                        "memory_degraded_reason": bundle.memory_degraded_reason,
                        "macro_context": bundle.macro_context,
                        "investor_profile": bundle.investor_profile,
                    },
                    "risk_gate": bundle.risk_gate,
                    "evidence_cards": [card.model_dump() for card in evidence_cards],
                    "citations": [citation.model_dump() for citation in citations],
                },
                response_payload=response_model.model_dump(mode="json"),
            )
        except TraceStoreUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "trace_store_unavailable",
                    "message": f"Analysis trace store unavailable: {exc}",
                },
            ) from exc

        return AnalysisComputation(
            response=response_model,
            bundle=bundle,
            horizon_forecasts=horizon_forecasts,
            recent_news=news.items[:6],
            evidence_cards=evidence_cards,
            citations=citations,
        )

    async def current_forecasts(self) -> AgentForecastsResponse:
        t0 = datetime.now(timezone.utc)
        snapshot, forecast_map, tool_trace = await self._gather_forecast_baseline()
        horizon_forecasts = [
            self._build_stable_horizon_forecast_card(
                horizon=horizon,
                forecast=forecast_map[horizon],
                snapshot=snapshot,
            )
            for horizon in ("24h", "7d", "30d")
        ]
        elapsed_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
        return AgentForecastsResponse(
            as_of=snapshot.as_of,
            market_status={
                "asset": snapshot.asset,
                "as_of": snapshot.as_of.isoformat(),
                "latest_price": snapshot.latest_price,
                "price_change_pct_1d": snapshot.price_change_pct_1d,
                "freshness_seconds": snapshot.freshness_seconds,
                "is_stale": snapshot.is_stale,
                "status": snapshot.status,
                "degraded_reason": snapshot.degraded_reason,
            },
            horizon_forecasts=horizon_forecasts,
            degradation_flags=_forecast_degradation_flags(snapshot=snapshot, forecast_map=forecast_map),
            timing_ms={
                "total": elapsed_ms,
                **{str(item["tool"]): int(item["elapsed_ms"]) for item in tool_trace},
            },
        )

    async def current_dashboard(self) -> AgentDashboardResponse:
        t0 = datetime.now(timezone.utc)
        snapshot, forecast_map, forecast_trace = await self._gather_forecast_baseline()
        horizon_forecasts = [
            self._build_stable_horizon_forecast_card(
                horizon=horizon,
                forecast=forecast_map[horizon],
                snapshot=snapshot,
            )
            for horizon in ("24h", "7d", "30d")
        ]
        indicators_started = datetime.now(timezone.utc)
        try:
            indicators = await self._toolbox.get_market_indicators()
            indicators_elapsed = int((datetime.now(timezone.utc) - indicators_started).total_seconds() * 1000)
            indicators_trace = self._tool_trace_entry(
                "get_market_indicators",
                indicators.model_dump(mode="json"),
                elapsed_ms=indicators_elapsed,
                status="degraded" if indicators.status != "ok" else "ok",
            )
            indicators_trace["source_status"] = indicators.status
            indicators_trace["degraded"] = indicators.status != "ok"
            indicators_trace["fallback_reason"] = indicators.degraded_reason
        except Exception as exc:
            indicators_elapsed = int((datetime.now(timezone.utc) - indicators_started).total_seconds() * 1000)
            indicators = _fallback_market_indicators(snapshot, f"{type(exc).__name__}:{exc}")
            indicators_trace = self._tool_trace_entry(
                "get_market_indicators",
                indicators.model_dump(mode="json"),
                elapsed_ms=indicators_elapsed,
                status="fallback",
                error=f"{type(exc).__name__}:{exc}",
            )
            indicators_trace["source_status"] = indicators.status
            indicators_trace["degraded"] = True
            indicators_trace["fallback_reason"] = indicators.degraded_reason

        history_started = datetime.now(timezone.utc)
        try:
            gold_history = await self._toolbox.get_gold_history()
            history_elapsed = int((datetime.now(timezone.utc) - history_started).total_seconds() * 1000)
            history_trace = self._tool_trace_entry(
                "get_gold_history",
                gold_history.model_dump(mode="json"),
                elapsed_ms=history_elapsed,
                status="ok" if not gold_history.source.startswith("synthetic_fallback") else "degraded",
            )
            history_trace["source_status"] = "ok"
            history_trace["degraded"] = gold_history.source.startswith("synthetic_fallback")
            history_trace["fallback_reason"] = None if not history_trace["degraded"] else gold_history.source
        except Exception as exc:
            history_elapsed = int((datetime.now(timezone.utc) - history_started).total_seconds() * 1000)
            gold_history = _fallback_gold_history(snapshot, f"{type(exc).__name__}:{exc}")
            history_trace = self._tool_trace_entry(
                "get_gold_history",
                gold_history.model_dump(mode="json"),
                elapsed_ms=history_elapsed,
                status="fallback",
                error=f"{type(exc).__name__}:{exc}",
            )
            history_trace["source_status"] = "degraded"
            history_trace["degraded"] = True
            history_trace["fallback_reason"] = gold_history.source

        news_started = datetime.now(timezone.utc)
        try:
            news = await self._toolbox.search_recent_news("黄金 美元 利率 ETF CFTC", limit=6)
            news_elapsed = int((datetime.now(timezone.utc) - news_started).total_seconds() * 1000)
            news_trace = self._tool_trace_entry(
                "search_dashboard_news",
                news,
                elapsed_ms=news_elapsed,
                status="degraded" if news.status != "ok" else "ok",
            )
        except Exception as exc:
            news_elapsed = int((datetime.now(timezone.utc) - news_started).total_seconds() * 1000)
            news = _fallback_recent_news("黄金市场")
            news_trace = self._tool_trace_entry(
                "search_dashboard_news",
                news,
                elapsed_ms=news_elapsed,
                status="fallback",
                error=f"{type(exc).__name__}:{exc}",
            )

        degradation_flags = _forecast_degradation_flags(snapshot=snapshot, forecast_map=forecast_map)
        if indicators.status != "ok":
            degradation_flags.append("market_indicators_degraded")
        if history_trace.get("degraded") or history_trace.get("status") == "fallback":
            degradation_flags.append("gold_history_degraded")
        if news.status != "ok":
            degradation_flags.append(f"news_{news.status}")
        degradation_flags = list(dict.fromkeys(degradation_flags))
        tool_trace = [*forecast_trace, indicators_trace, history_trace, news_trace]
        elapsed_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
        degraded_tools = [item["tool"] for item in tool_trace if item.get("degraded") or item.get("status") == "fallback"]
        quality_status = "degraded" if degradation_flags or degraded_tools else "ok"
        source_health = _build_dashboard_source_health(
            snapshot=snapshot,
            indicators=indicators,
            news=news,
            news_expected_lag_seconds=self._cfg.news_stale_after_seconds,
        )
        return AgentDashboardResponse(
            as_of=snapshot.as_of,
            market_status={
                "asset": snapshot.asset,
                "as_of": snapshot.as_of.isoformat(),
                "latest_price": snapshot.latest_price,
                "price_change_pct_1d": snapshot.price_change_pct_1d,
                "freshness_seconds": snapshot.freshness_seconds,
                "is_stale": snapshot.is_stale,
                "status": snapshot.status,
                "degraded_reason": snapshot.degraded_reason,
            },
            horizon_forecasts=horizon_forecasts,
            indicator_groups=indicators.groups,
            gold_history=gold_history,
            recent_news=news.items[:6],
            citations=[item.model_dump(mode="json") for item in indicators.citations],
            source_health=source_health,
            data_quality={
                "status": quality_status,
                "degraded_tools": degraded_tools,
                "freshness_seconds": snapshot.freshness_seconds,
                "indicator_status": indicators.status,
                "news_status": news.status,
            },
            degradation_flags=degradation_flags,
            timing_ms={
                "total": elapsed_ms,
                **{str(item["tool"]): int(item["elapsed_ms"]) for item in tool_trace},
            },
        )

    def _forecast_trace_entry(
        self,
        horizon: PublicHorizon,
        payload: Dict[str, Any],
        *,
        elapsed_ms: int,
        status: str,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        trace = self._tool_trace_entry(
            f"get_quant_forecast_{horizon}",
            payload,
            elapsed_ms=elapsed_ms,
            status=status,
            error=error,
        )
        trace["status"] = "degraded" if trace["degraded"] and status == "ok" else status
        return trace

    async def _gather_quant_forecasts(
        self, horizons: Sequence[PublicHorizon]
    ) -> Tuple[Dict[PublicHorizon, Dict[str, Any]], List[Dict[str, Any]]]:
        batch_method = getattr(self._toolbox, "get_quant_forecasts", None)
        if callable(batch_method):
            started = datetime.now(timezone.utc)
            try:
                payload = await batch_method(list(horizons))
                elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
                forecast_map: Dict[PublicHorizon, Dict[str, Any]] = {}
                traces: List[Dict[str, Any]] = []
                for horizon in horizons:
                    forecast = payload.get(horizon)
                    if forecast is None:
                        forecast = _fallback_quant_forecast("batch_missing_horizon")
                        status = "fallback"
                        error = "batch_missing_horizon"
                    else:
                        status = "ok"
                        error = None
                    forecast_map[horizon] = forecast
                    traces.append(
                        self._forecast_trace_entry(
                            horizon,
                            forecast,
                            elapsed_ms=elapsed_ms,
                            status=status,
                            error=error,
                        )
                    )
                return forecast_map, traces
            except Exception as exc:
                elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
                reason = f"{type(exc).__name__}:{exc}"
                forecast_map = {horizon: _fallback_quant_forecast(reason) for horizon in horizons}
                traces = [
                    self._forecast_trace_entry(
                        horizon,
                        forecast_map[horizon],
                        elapsed_ms=elapsed_ms,
                        status="fallback",
                        error=reason,
                    )
                    for horizon in horizons
                ]
                return forecast_map, traces

        timed = await asyncio.gather(
            *[
                self._timed_optional_tool(
                    f"get_quant_forecast_{horizon}",
                    self._toolbox.get_quant_forecast(horizon),
                    lambda exc: _fallback_quant_forecast(f"{type(exc).__name__}:{exc}"),
                )
                for horizon in horizons
            ]
        )
        forecast_map = {horizon: timed[idx][0] for idx, horizon in enumerate(horizons)}
        traces = [item[1] for item in timed]
        return forecast_map, traces

    async def _gather(
        self, req: AgentAnalyzeRequest, *, evidence_query: str, news_query: str
    ) -> Tuple[MarketSnapshotResponse, Dict[PublicHorizon, Dict[str, Any]], RecentNewsResponse, HistoricalEventsLookup, List[Dict[str, Any]]]:
        try:
            timed_snapshot, timed_forecasts, timed_news, timed_rag = await asyncio.gather(
                self._timed_tool("get_market_snapshot", self._toolbox.get_market_snapshot()),
                self._gather_quant_forecasts(("24h", "7d", "30d")),
                self._timed_optional_tool(
                    "search_recent_news",
                    self._toolbox.search_recent_news(news_query, limit=6),
                    lambda exc: _fallback_recent_news(evidence_query),
                ),
                self._timed_optional_tool(
                    "retrieve_historical_events",
                    self._toolbox.retrieve_historical_events(evidence_query, top_k=3),
                    lambda exc: HistoricalEventsLookup(
                        items=[],
                        status="degraded",
                        degraded_reason=f"memory_lookup_failed:{type(exc).__name__}:{exc}",
                        source_freshness_seconds=None,
                    ),
                ),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "upstream_unavailable",
                    "message": f"Upstream tool failed: {type(exc).__name__}: {exc}",
                },
            ) from exc
        forecast_map, forecast_trace = timed_forecasts
        tool_trace = [
            timed_snapshot[1],
            *forecast_trace,
            timed_news[1],
            timed_rag[1],
        ]
        snapshot, news, memory_lookup = timed_snapshot[0], timed_news[0], _normalize_memory_lookup(timed_rag[0])
        return snapshot, forecast_map, news, memory_lookup, tool_trace

    async def _gather_forecast_baseline(self) -> Tuple[MarketSnapshotResponse, Dict[PublicHorizon, Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            timed_snapshot, timed_forecasts = await asyncio.gather(
                self._timed_tool("get_market_snapshot", self._toolbox.get_market_snapshot()),
                self._gather_quant_forecasts(("24h", "7d", "30d")),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "upstream_unavailable",
                    "message": f"Forecast baseline failed: {type(exc).__name__}: {exc}",
                },
            ) from exc

        forecast_map, forecast_trace = timed_forecasts
        tool_trace = [
            timed_snapshot[1],
            *forecast_trace,
        ]
        return timed_snapshot[0], forecast_map, tool_trace

    def _tool_trace_entry(self, name: str, payload: Any, *, elapsed_ms: int, status: str, error: Optional[str] = None) -> Dict[str, Any]:
        source_status = "ok"
        degraded = False
        fallback_reason = None
        source_freshness_seconds = None

        if isinstance(payload, MarketSnapshotResponse):
            source_status = payload.status
            degraded = payload.status != "ok"
            fallback_reason = payload.degraded_reason
            source_freshness_seconds = payload.source_freshness_seconds
        elif isinstance(payload, RecentNewsResponse):
            source_status = payload.status
            degraded = payload.status != "ok"
            fallback_reason = payload.degraded_reason
            source_freshness_seconds = payload.source_freshness_seconds
        elif isinstance(payload, HistoricalEventsLookup):
            source_status = payload.status
            degraded = payload.status != "ok"
            fallback_reason = payload.degraded_reason
            source_freshness_seconds = payload.source_freshness_seconds
        elif isinstance(payload, dict):
            source_status = str(payload.get("service_status", "ok"))
            degraded = source_status != "ok"
            fallback_reason = payload.get("reason") or payload.get("degraded_reason")
            source_freshness_seconds = payload.get("source_freshness_seconds")

        trace = {
            "tool": name,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "degraded": degraded,
            "source_status": source_status,
            "fallback_reason": fallback_reason,
            "source_freshness_seconds": source_freshness_seconds,
        }
        if isinstance(payload, dict) and "model_status" in payload:
            trace["model_status"] = payload.get("model_status")
            trace["model_loaded"] = payload.get("model_loaded")
            trace["model_checkpoint_path"] = payload.get("model_checkpoint_path")
        if isinstance(payload, dict) and "forecast_basis" in payload:
            trace["forecast_basis"] = payload.get("forecast_basis")
        if error is not None:
            trace["error"] = error
        return trace

    async def _timed_tool(self, name: str, awaitable):
        started = datetime.now(timezone.utc)
        try:
            payload = await awaitable
            elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
            trace_status = "degraded" if self._tool_trace_entry(name, payload, elapsed_ms=elapsed_ms, status="ok")["degraded"] else "ok"
            return payload, self._tool_trace_entry(name, payload, elapsed_ms=elapsed_ms, status=trace_status)
        except Exception as exc:
            elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
            raise RuntimeError(f"{name}:{type(exc).__name__}:{exc}") from exc

    async def _timed_optional_tool(self, name: str, awaitable, fallback_factory):
        started = datetime.now(timezone.utc)
        try:
            payload = await awaitable
            elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
            trace = self._tool_trace_entry(name, payload, elapsed_ms=elapsed_ms, status="ok")
            trace["status"] = "degraded" if trace["degraded"] else "ok"
            return payload, trace
        except Exception as exc:
            elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
            payload = fallback_factory(exc)
            return payload, self._tool_trace_entry(
                name,
                payload,
                elapsed_ms=elapsed_ms,
                status="fallback",
                error=f"{type(exc).__name__}:{exc}",
            )

    def _derive_news_sentiment(self, req: AgentAnalyzeRequest, news: RecentNewsResponse) -> float:
        scores = [item.sentiment_score for item in news.items[:5]]
        if req.optional_news_text:
            scores.append(self._sentiment_scorer.score(req.optional_news_text))
        return _clamp(_mean(scores), -1.0, 1.0)

    def _memory_average(self, horizon: PublicHorizon, rag_events: List[RagEventItem]) -> float:
        values: List[Optional[float]] = []
        for event in rag_events:
            if horizon == "24h":
                values.append(event.gold_t1_return)
            elif horizon == "7d":
                values.append(event.gold_t7_return)
            else:
                proxy = None
                if event.gold_t7_return is not None:
                    proxy = event.gold_t7_return * 1.6
                elif event.gold_t1_return is not None:
                    proxy = event.gold_t1_return * 2.8
                values.append(proxy)
        return _mean(values)

    def _compute_conflict(
        self,
        *,
        quant_direction: int,
        news_sentiment: float,
        memory_avg: float,
        technical_state: str,
        macro_signal: int,
    ) -> Tuple[int, bool]:
        signals: List[int] = []
        if quant_direction != 0:
            signals.append(quant_direction)
        if news_sentiment > 0.08:
            signals.append(1)
        elif news_sentiment < -0.08:
            signals.append(-1)
        if memory_avg > 0.002:
            signals.append(1)
        elif memory_avg < -0.002:
            signals.append(-1)
        if technical_state == "bullish":
            signals.append(1)
        elif technical_state == "bearish":
            signals.append(-1)
        if macro_signal != 0:
            signals.append(macro_signal)
        if len(signals) <= 1:
            return 0, False

        positive = len([x for x in signals if x > 0])
        negative = len([x for x in signals if x < 0])
        conflict_score = min(positive, negative)
        return conflict_score, conflict_score >= 2

    def _evaluate_horizon_outlook(
        self,
        *,
        horizon: PublicHorizon,
        forecast: Dict[str, Any],
        snapshot: MarketSnapshotResponse,
        news: RecentNewsResponse,
        news_sentiment: float,
        rag_events: List[RagEventItem],
        memory_lookup: HistoricalEventsLookup,
        macro_context: Dict[str, Any],
        risk_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        probability = _safe_float(forecast.get("probability")) or 0.5
        xgboost_probability = _safe_float(forecast.get("xgboost_probability"))
        quant_direction = int(forecast.get("direction_prediction", 0) or 0)
        vix_value = next((item.price for item in snapshot.instruments if item.symbol == "VIX"), None)
        memory_avg = self._memory_average(horizon, rag_events)
        conflict_score, has_conflict = self._compute_conflict(
            quant_direction=quant_direction,
            news_sentiment=news_sentiment,
            memory_avg=memory_avg,
            technical_state=snapshot.feature_summary.technical_state,
            macro_signal=int(macro_context.get("macro_signal", 0)),
        )
        has_degraded_inputs = (
            _snapshot_uses_fallback(snapshot)
            or _forecast_is_degraded(forecast)
            or _news_uses_fallback(news)
            or news.status != "ok"
        )
        basis = _forecast_basis(forecast)
        low_confidence_threshold = {"24h": 0.56, "7d": 0.58, "30d": 0.6}[horizon]
        is_low_confidence = probability < low_confidence_threshold
        is_high_risk = bool(
            snapshot.is_stale
            or has_degraded_inputs
            or bool(risk_profile.get("force_observation"))
            or (vix_value is not None and vix_value >= self._cfg.vix_circuit_breaker_threshold)
        )

        stance: SummaryStance = "中性"
        action: SummaryAction = "观望"
        confidence_band: ConfidenceBand = "低"
        if is_high_risk or has_conflict or is_low_confidence:
            stance = "高风险观望"
            action = "观望"
            confidence_band = "低"
        elif quant_direction > 0:
            stance = "偏多"
            confidence_band = "高" if probability >= 0.67 and basis == "ensemble_model" else "中"
            action = "小仓试探" if risk_profile["profile"] == "conservative" else "分批布局"
        elif quant_direction < 0:
            stance = "偏空"
            confidence_band = "高" if probability >= 0.67 and basis == "ensemble_model" else "中"
            action = "降低暴露"

        horizon_label = {"24h": "T+1", "7d": "T+7", "30d": "T+30"}[horizon]
        if basis == "degraded_fallback":
            basis_reason = f"{horizon_label} 量化暂不可用，当前已退回保守中性判断。"
        elif risk_profile.get("force_observation"):
            basis_reason = f"{horizon_label} 方向基线存在，但完整问卷触发风险门控，本轮只能观望。"
        elif basis == "heuristic_proxy":
            basis_reason = f"{horizon_label} 当前使用代理预测，主要参考趋势、美元、利率和自动抓取的新闻环境。"
        else:
            basis_reason = f"{horizon_label} 量化方向为 {'偏多' if quant_direction > 0 else '偏空' if quant_direction < 0 else '中性'}，概率约 {probability * 100:.1f}%。"

        if horizon == "30d":
            horizon_reason = "30 天更看中期趋势与宏观环境，因此结论会比短线更慢、更偏参考。"
        elif horizon == "7d":
            horizon_reason = (
                f"历史相似事件在 7 天口径的均值表现约 {memory_avg * 100:+.2f}%，可帮助判断冲击是否延续。"
                if rag_events
                else "7 天视角会同时参考新闻延续性和趋势结构。"
            )
        else:
            horizon_reason = (
                f"历史相似事件在 1 天口径的均值表现约 {memory_avg * 100:+.2f}%，更适合短线参考。"
                if rag_events
                else "24 小时视角更容易受新闻和美元短线波动影响。"
            )

        return {
            "probability": probability,
            "xgboost_probability": xgboost_probability,
            "quant_direction": quant_direction,
            "vix_value": vix_value,
            "conflict_score": conflict_score,
            "has_conflict": has_conflict,
            "has_degraded_inputs": has_degraded_inputs,
            "is_high_risk": is_high_risk,
            "is_low_confidence": is_low_confidence,
            "stance": stance,
            "action": action,
            "confidence_band": confidence_band,
            "basis": basis,
            "reasons": [
                basis_reason,
                macro_context["dollar_message"],
                macro_context["news_message"],
                horizon_reason,
            ][:4],
        }

    def _build_stable_horizon_forecast_card(
        self,
        *,
        horizon: PublicHorizon,
        forecast: Dict[str, Any],
        snapshot: MarketSnapshotResponse,
    ) -> HorizonForecastCard:
        probability = _safe_float(forecast.get("probability")) or 0.5
        quant_direction = int(forecast.get("direction_prediction", 0) or 0)
        basis = _forecast_basis(forecast)
        vix_value = next((item.price for item in snapshot.instruments if item.symbol == "VIX"), None)
        is_high_risk = bool(
            snapshot.is_stale
            or _snapshot_uses_fallback(snapshot)
            or _forecast_is_degraded(forecast)
            or (vix_value is not None and vix_value >= self._cfg.vix_circuit_breaker_threshold)
        )
        low_confidence_threshold = {"24h": 0.56, "7d": 0.58, "30d": 0.6}[horizon]

        stance: SummaryStance = "中性"
        action: SummaryAction = "观望"
        confidence_band: ConfidenceBand = "低"
        if is_high_risk or probability < low_confidence_threshold:
            stance = "高风险观望" if is_high_risk else "中性"
            action = "观望"
            confidence_band = "低"
        elif quant_direction > 0:
            stance = "偏多"
            action = "小仓试探"
            confidence_band = "高" if probability >= 0.67 and basis == "ensemble_model" else "中"
        elif quant_direction < 0:
            stance = "偏空"
            action = "降低暴露"
            confidence_band = "高" if probability >= 0.67 and basis == "ensemble_model" else "中"

        horizon_label = {"24h": "T+1", "7d": "T+7", "30d": "T+30"}[horizon]
        if basis == "degraded_fallback":
            basis_reason = f"{horizon_label} 量化预测暂不可用，当前只保留保守占位。"
        elif basis == "heuristic_proxy":
            basis_reason = f"{horizon_label} 使用代理预测，未读取用户问题文本。"
        else:
            basis_reason = f"{horizon_label} 稳定量化方向为 {'偏多' if quant_direction > 0 else '偏空' if quant_direction < 0 else '中性'}，概率约 {probability * 100:.1f}%。"

        market_reason = (
            f"市场快照截至 {snapshot.as_of.isoformat()}，XAUUSD 最新价约 {snapshot.latest_price:.2f}。"
        )
        freshness_reason = (
            "行情快照或模型处于降级状态，因此该预测需要按低置信度处理。"
            if is_high_risk
            else "该卡只使用行情快照和量化模型输出，不随聊天输入改写。"
        )
        horizon_reason = {
            "24h": "T+1 用于短线方向基线，适合和即时新闻解释分开阅读。",
            "7d": "T+7 用于一周方向基线，避免单条问题改变市场预测。",
            "30d": "T+30 用于中期参考，不等同于独立长期交易建议。",
        }[horizon]

        return HorizonForecastCard(
            horizon=horizon,
            stance=stance,
            confidence_band=confidence_band,
            action=action,
            probability=probability,
            basis=basis,
            model_status=str(forecast.get("model_status", "unknown")),
            model_loaded=bool(forecast.get("model_loaded", False)),
            model_checkpoint_path=forecast.get("model_checkpoint_path"),
            reasons=[basis_reason, market_reason, freshness_reason, horizon_reason],
        )

    def _build_horizon_forecast_card(
        self,
        *,
        horizon: PublicHorizon,
        forecast: Dict[str, Any],
        snapshot: MarketSnapshotResponse,
        news: RecentNewsResponse,
        news_sentiment: float,
        rag_events: List[RagEventItem],
        memory_lookup: HistoricalEventsLookup,
        macro_context: Dict[str, Any],
        risk_profile: Dict[str, Any],
    ) -> HorizonForecastCard:
        outlook = self._evaluate_horizon_outlook(
            horizon=horizon,
            forecast=forecast,
            snapshot=snapshot,
            news=news,
            news_sentiment=news_sentiment,
            rag_events=rag_events,
            memory_lookup=memory_lookup,
            macro_context=macro_context,
            risk_profile=risk_profile,
        )
        return HorizonForecastCard(
            horizon=horizon,
            stance=outlook["stance"],
            confidence_band=outlook["confidence_band"],
            action=outlook["action"],
            probability=float(outlook["probability"]),
            basis=outlook["basis"],
            model_status=str(forecast.get("model_status", "unknown")),
            model_loaded=bool(forecast.get("model_loaded", False)),
            model_checkpoint_path=forecast.get("model_checkpoint_path"),
            reasons=outlook["reasons"],
        )

    def _build_citations(self, bundle: AnalysisBundle) -> List[CitationItem]:
        citations: List[CitationItem] = []
        snapshot = bundle.snapshot
        if _snapshot_uses_fallback(snapshot):
            market_excerpt = "市场快照暂不可用，当前使用本地样本数据维持演示链路，系统已自动切换为保守模式。"
        else:
            market_excerpt = (
                f"黄金现价 {snapshot.latest_price:.2f}，1日变动 "
                f"{(snapshot.price_change_pct_1d or 0.0) * 100:+.2f}%，"
                f"波动状态 {snapshot.feature_summary.volatility_regime}。"
            )
        citations.append(
            CitationItem(
                id="cit-market",
                label="市场快照",
                source_type="market_snapshot",
                excerpt=market_excerpt,
            )
        )
        citations.append(
            CitationItem(
                id="cit-quant",
                label="量化预测",
                source_type="quant_forecast",
                excerpt=(
                    "量化引擎暂不可用，系统已用中性占位并降级为保守建议。"
                    if _forecast_is_degraded(bundle.forecast)
                    else (
                        f"{bundle.horizon} 当前使用代理预测，概率 {((bundle.quant_probability or 0.0) * 100):.1f}%，更适合作为中期参考。"
                        if _forecast_basis(bundle.forecast) == "heuristic_proxy"
                        else (
                            f"方向信号 {bundle.forecast.get('direction_prediction')}，"
                            f"集成概率 {bundle.quant_probability if bundle.quant_probability is not None else 'N/A'}，"
                            f"XGBoost 概率 {bundle.xgboost_probability if bundle.xgboost_probability is not None else 'N/A'}。"
                        )
                    )
                ),
            )
        )
        if bundle.news.items:
            top_news = bundle.news.items[0]
            citations.append(
                CitationItem(
                    id="cit-news-1",
                    label=f"近期新闻：{top_news.source}",
                    source_type="recent_news",
                    excerpt=(
                        f"{top_news.title}。摘要：{top_news.summary}"
                        if bundle.news.status == "ok"
                        else f"{top_news.title}。当前新闻源处于 {bundle.news.status} 状态：{bundle.news.degraded_reason or '无'}。"
                    ),
                    url=top_news.url,
                )
            )
        if bundle.rag_events:
            first = bundle.rag_events[0]
            citations.append(
                CitationItem(
                    id="cit-memory",
                    label="历史相似事件",
                    source_type="historical_analogs",
                    excerpt=(
                        f"{first.headline}；T+1 {((first.gold_t1_return or 0.0) * 100):+.2f}%，"
                        f"T+7 {((first.gold_t7_return or 0.0) * 100):+.2f}%。"
                    ),
                )
            )
        elif bundle.memory_status != "ok":
            citations.append(
                CitationItem(
                    id="cit-memory",
                    label="历史相似事件",
                    source_type="historical_analogs",
                    excerpt=_memory_unavailable_copy(bundle.memory_status, bundle.memory_degraded_reason),
                )
            )
        citations.append(
            CitationItem(
                id="cit-macro",
                label="宏观语境",
                source_type="macro_context",
                excerpt=f"{bundle.macro_context['dollar_message']} {bundle.macro_context['rate_message']} {bundle.macro_context['news_message']}",
            )
        )
        citations.append(
            CitationItem(
                id="cit-risk",
                label="用户风险画像",
                source_type="risk_profile",
                excerpt=(
                    f"{bundle.risk_profile['description']} 风险门控："
                    f"{bundle.risk_gate.get('level')} / {', '.join(bundle.risk_gate.get('notes', []))}"
                ),
            )
        )
        return citations

    def _build_evidence_cards(
        self, bundle: AnalysisBundle, citations: List[CitationItem]
    ) -> List[EvidenceCard]:
        quant_dir = bundle.quant_direction
        bullish = quant_dir > 0
        cards: List[EvidenceCard] = []

        tech_direction = "supportive" if bundle.snapshot.feature_summary.technical_state == ("bullish" if bullish else "bearish") else "neutral"
        cards.append(
            EvidenceCard(
                id="ev-market",
                title="市场状态",
                signal_type="market",
                takeaway=(
                    f"当前技术面为 {bundle.snapshot.feature_summary.technical_state}，"
                    f"波动率状态为 {bundle.snapshot.feature_summary.volatility_regime}。"
                ),
                direction=tech_direction,
                citation_ids=["cit-market"],
            )
        )

        quant_direction = "偏多" if bullish else "偏空" if quant_dir < 0 else "中性"
        cards.append(
            EvidenceCard(
                id="ev-quant",
                title="量化预测",
                signal_type="quant",
                takeaway=(
                    "量化引擎当前暂不可用，系统已按中性概率处理并自动收紧建议。"
                    if _forecast_is_degraded(bundle.forecast)
                    else (
                        f"{bundle.snapshot.asset} 的 {bundle.horizon} 当前采用代理预测，概率 {((bundle.quant_probability or 0.0) * 100):.1f}%。"
                        if _forecast_basis(bundle.forecast) == "heuristic_proxy"
                        else (
                        f"{bundle.snapshot.asset} 的 {bundle.horizon} 量化方向为 {quant_direction}，"
                        f"集成概率 {((bundle.quant_probability or 0.0) * 100):.1f}%。"
                        )
                    )
                ),
                direction="neutral" if _forecast_is_degraded(bundle.forecast) else "supportive",
                citation_ids=["cit-quant"],
            )
        )

        news_direction = "neutral"
        if bundle.news_sentiment > 0.08:
            news_direction = "supportive" if bullish else "contradictory"
        elif bundle.news_sentiment < -0.08:
            news_direction = "supportive" if not bullish else "contradictory"
        headline = bundle.news.items[0].title if bundle.news.items else "暂无高置信新闻"
        cards.append(
            EvidenceCard(
                id="ev-news",
                title="新闻与情绪",
                signal_type="news",
                takeaway=f"最近新闻主线：{headline}。综合新闻情绪得分 {bundle.news_sentiment:+.2f}。",
                direction=news_direction,
                citation_ids=["cit-news-1"] if any(c.id == "cit-news-1" for c in citations) else ["cit-macro"],
            )
        )

        memory_avg = _mean(
            [
                event.gold_t1_return if bundle.horizon == "24h" else event.gold_t7_return
                for event in bundle.rag_events
            ]
        )
        memory_direction = "neutral"
        if memory_avg > 0.002:
            memory_direction = "supportive" if bullish else "contradictory"
        elif memory_avg < -0.002:
            memory_direction = "supportive" if not bullish else "contradictory"
        cards.append(
            EvidenceCard(
                id="ev-memory",
                title="历史相似事件",
                signal_type="memory",
                takeaway=(
                    _memory_unavailable_copy(bundle.memory_status, bundle.memory_degraded_reason)
                    if bundle.memory_status != "ok"
                    else (
                        "历史相似事件均值表现 "
                        f"{memory_avg * 100:+.2f}%（按 {'T+1' if bundle.horizon == '24h' else 'T+7'} 口径），"
                        "可用于判断新闻冲击是否容易延续。"
                    )
                ),
                direction="neutral" if bundle.memory_status != "ok" else memory_direction,
                citation_ids=["cit-memory"] if any(c.id == "cit-memory" for c in citations) else ["cit-macro"],
            )
        )

        cards.append(
            EvidenceCard(
                id="ev-macro",
                title="宏观语境",
                signal_type="macro",
                takeaway=f"{bundle.macro_context['dollar_message']} {bundle.macro_context['rate_message']}",
                direction="contradictory" if bundle.has_conflict else "neutral",
                citation_ids=["cit-macro"],
            )
        )

        cards.append(
            EvidenceCard(
                id="ev-risk",
                title="用户风险画像",
                signal_type="risk",
                takeaway=(
                    f"{bundle.risk_profile['label']}：{bundle.risk_profile['description']} "
                    f"问卷门控等级 {bundle.risk_gate.get('level')}。"
                ),
                direction="neutral",
                citation_ids=["cit-risk"],
            )
        )
        return cards

    def _build_draft_narrative(self, req: AgentAnalyzeRequest, bundle: AnalysisBundle) -> NarrativeOutput:
        stance: SummaryStance = "中性"
        action: SummaryAction = "观望"
        confidence_band: ConfidenceBand = "低"

        bullish = bundle.quant_direction > 0
        if bundle.risk_gate.get("force_observation") or bundle.is_high_risk or bundle.has_conflict or bundle.is_low_confidence:
            stance = "高风险观望"
            action = "观望"
            confidence_band = "低"
        else:
            if bullish:
                stance = "偏多"
                confidence_band = "高" if (bundle.quant_probability or 0.0) >= 0.66 else "中"
                action = "小仓试探" if req.risk_profile == "conservative" else "分批布局"
            elif bundle.quant_direction < 0:
                stance = "偏空"
                confidence_band = "高" if (bundle.quant_probability or 0.0) >= 0.66 else "中"
                action = "降低暴露"
            else:
                stance = "中性"
                action = "观望"
                confidence_band = "低"

        reasons = [
            (
                f"完整问卷触发风险画像门控：{' '.join(bundle.risk_gate.get('notes', []))}"
                if bundle.risk_gate.get("force_observation")
                else ""
            ),
            (
                "量化引擎当前不可用，系统已切换为保守中性处理。"
                if _forecast_is_degraded(bundle.forecast)
                else (
                    f"{bundle.horizon} 当前采用代理预测，主要参考趋势、美元、利率和自动抓取的新闻环境。"
                    if _forecast_basis(bundle.forecast) == "heuristic_proxy"
                    else f"量化层给出的主方向是 {'偏多' if bundle.quant_direction > 0 else '偏空' if bundle.quant_direction < 0 else '中性'}，概率约 {(bundle.quant_probability or 0.0) * 100:.1f}%。"
                )
            ),
            bundle.macro_context["dollar_message"],
            bundle.macro_context["news_message"],
        ]
        reasons = [reason for reason in reasons if reason][:4]
        invalidators = [
            "如果美元和实际利率同步快速走强，当前观点需要重新评估。",
            "如果接下来 1-2 个交易时段新闻方向反转，历史类比可能失效。",
            "如果 VIX 升到 30 以上或数据明显陈旧，应立即降级为观望。",
        ]
        disclaimer = "本内容仅用于帮助理解黄金市场，不构成个性化投资建议，也不替代你自己的风险决策。"

        if bundle.risk_gate.get("force_observation"):
            risk_banner = RiskBanner(
                level="high",
                title="问卷风险门控",
                message="完整风险问卷显示本轮暴露与承受能力不匹配，系统只输出观望和风险边界。",
            )
        elif bundle.has_degraded_inputs:
            risk_banner = RiskBanner(
                level="high",
                title="降级模式",
                message="部分实时数据或模型暂不可用，系统已自动切换为保守模式，只保留解释性建议。",
            )
        elif bundle.is_high_risk:
            risk_banner = RiskBanner(
                level="high",
                title="高风险环境",
                message="当前波动率或数据新鲜度不满足稳健建议条件，系统已自动降级为观望。",
            )
        elif bundle.has_conflict:
            risk_banner = RiskBanner(
                level="high",
                title="证据冲突",
                message="新闻、技术面和历史类比没有形成一致结论，强结论风险较高。",
            )
        elif bundle.is_low_confidence:
            risk_banner = RiskBanner(
                level="medium",
                title="置信度偏低",
                message="量化胜率不足，当前更适合观察关键价位和下一条宏观催化。",
            )
        else:
            risk_banner = RiskBanner(
                level="low" if confidence_band == "高" else "medium",
                title="风险提醒",
                message="即使方向偏明确，也应分步行动，并在关键宏观数据前控制仓位节奏。",
            )

        follow_up_questions = [
            "如果你已经持有黄金仓位，我可以按你的风险偏好重写成持仓建议。",
            "如果你想比较 T+1、T+7、T+30 哪个周期分歧最大，我可以直接帮你解释。",
            "如果你想看这次判断最容易失效的情景，我可以单独展开风险清单。",
        ]

        return NarrativeOutput(
            summary_card=SummaryCard(
                stance=stance,
                horizon=req.horizon,
                confidence_band=confidence_band,
                action=action,
                reasons=reasons,
                invalidators=invalidators,
                disclaimer=disclaimer,
            ),
            risk_banner=risk_banner,
            follow_up_questions=follow_up_questions,
        )


def _legacy_decision_from_analyze(resp: AgentAnalyzeResponse) -> AgentDecision:
    stance = resp.summary_card.stance
    if stance == "偏多" and resp.summary_card.action in {"小仓试探", "分批布局"}:
        action = "BUY"
    elif stance == "偏空" and resp.summary_card.action == "降低暴露":
        action = "SELL"
    else:
        action = "HOLD"
    return AgentDecision(
        action=action,
        confidence=_confidence_to_score(resp.summary_card.confidence_band),
        horizon="T+1" if resp.summary_card.horizon == "24h" else "T+7",
        reasoning_summary=resp.summary_card.reasons[0],
        risk_warning=resp.risk_banner.message,
    )


def _legacy_risk_result(resp: AgentAnalyzeResponse, vix_value: Optional[float], threshold: float) -> RiskResult:
    if resp.summary_card.action == "观望" and resp.risk_banner.level == "high":
        decision: RiskResult = {
            "decision": "REJECTED",
            "executed_position": 0.0,
            "current_vix": vix_value,
            "vix_threshold": threshold,
            "notes": resp.risk_banner.message,
        }
        return decision
    return {
        "decision": "PASS",
        "executed_position": 0.0,
        "current_vix": vix_value,
        "vix_threshold": threshold,
        "notes": resp.risk_banner.message,
    }


def _health_url(service_url: str) -> str:
    base = service_url.split("/api/", 1)[0].rstrip("/")
    return f"{base}/health/ready"


def create_app(
    *,
    toolbox: Optional[HttpResearchToolbox] = None,
    narrator: Optional[OpenAINarrator] = None,
    sentiment_scorer: Optional[BaseSentimentScorer] = None,
    trace_store: Optional[AgentTraceStore] = None,
    http_client: Optional[httpx.AsyncClient] = None,
) -> FastAPI:
    tool_timeout_seconds = float(_env("AGENT_TOOL_TIMEOUT_SECONDS", "35.0"))
    tool_connect_timeout_seconds = float(_env("AGENT_TOOL_CONNECT_TIMEOUT_SECONDS", "1.5"))
    app_env = _env("APP_ENV", "development").lower()
    cfg = AgentGatewayConfig(
        forecast_url=_env("FORECAST_URL", "http://localhost:8010/api/v1/forecast"),
        memory_url=_env("MEMORY_URL", "http://localhost:8012/api/v1/memory/search"),
        market_snapshot_url=_env("MARKET_SNAPSHOT_URL", "http://localhost:8014/api/v1/market/snapshot/latest"),
        market_indicators_url=_env("MARKET_INDICATORS_URL", "http://localhost:8014/api/v1/market/indicators/current"),
        market_history_url=_env("MARKET_HISTORY_URL", "http://localhost:8014/api/v1/market/gold/history"),
        recent_news_url=_env("RECENT_NEWS_URL", "http://localhost:8016/api/v1/news/recent"),
        default_model=_env("AGENT_DEFAULT_MODEL", "gpt-5.4-mini"),
        complex_model=_env("AGENT_COMPLEX_MODEL", "gpt-5.4"),
        vix_circuit_breaker_threshold=float(_env("VIX_CIRCUIT_BREAKER_THRESHOLD", "30")),
        stale_after_seconds=int(_env("MARKET_STALE_AFTER_SECONDS", "180")),
        news_stale_after_seconds=int(_env("NEWS_STALE_AFTER_SECONDS", "300")),
    )
    database_url = _env("DATABASE_URL", "postgresql://localhost/postgres")
    public_keys_raw = os.environ.get("AGENT_PUBLIC_API_KEYS")
    internal_keys_raw = os.environ.get("AGENT_INTERNAL_API_KEYS")
    if app_env != "development" and (not public_keys_raw or not internal_keys_raw):
        raise RuntimeError("AGENT_PUBLIC_API_KEYS and AGENT_INTERNAL_API_KEYS must be set outside development.")
    public_api_keys = _split_csv(public_keys_raw or "dev-public-key")
    internal_api_keys = _split_csv(internal_keys_raw or "dev-internal-key")
    if app_env != "development" and (
        "dev-public-key" in public_api_keys or "dev-internal-key" in internal_api_keys
    ):
        raise RuntimeError("Default development API keys are not allowed outside development.")
    analyze_rate_limit_per_minute = int(_env("AGENT_ANALYZE_RATE_LIMIT_PER_MINUTE", "60"))
    analyze_rate_limit_window_seconds = int(_env("AGENT_ANALYZE_RATE_LIMIT_WINDOW_SECONDS", "60"))
    allow_trace_memory_fallback = (
        os.environ.get("AGENT_ALLOW_TRACE_MEMORY_FALLBACK", "1" if app_env == "development" else "0") != "0"
    )
    trace_memory_ttl_seconds = int(_env("AGENT_TRACE_MEMORY_TTL_SECONDS", "3600"))
    trace_memory_max_items = int(_env("AGENT_TRACE_MEMORY_MAX_ITEMS", "200"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        own_http = http_client is None
        http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(tool_timeout_seconds, connect=tool_connect_timeout_seconds)
        )
        app.state.http = http
        app.state.cfg = cfg
        app.state.uses_injected_toolbox = toolbox is not None
        app.state.toolbox = toolbox or HttpResearchToolbox(http, cfg)
        app.state.narrator = narrator or OpenAINarrator(cfg)
        app.state.sentiment_scorer = sentiment_scorer or KeywordSentimentScorer()
        app.state.authorizer = ApiKeyAuthorizer(public_keys=public_api_keys, internal_keys=internal_api_keys)
        app.state.rate_limiter = SlidingWindowRateLimiter(
            limit=analyze_rate_limit_per_minute,
            window_seconds=analyze_rate_limit_window_seconds,
        )
        app.state.trace_store = trace_store or AgentTraceStore(
            database_url,
            allow_memory_fallback=allow_trace_memory_fallback,
            memory_ttl_seconds=trace_memory_ttl_seconds,
            memory_max_items=trace_memory_max_items,
        )
        if hasattr(app.state.trace_store, "startup"):
            await app.state.trace_store.startup()
        app.state.analysis_service = AgentAnalysisService(
            toolbox=app.state.toolbox,
            narrator=app.state.narrator,
            sentiment_scorer=app.state.sentiment_scorer,
            trace_store=app.state.trace_store,
            cfg=cfg,
        )
        yield
        if own_http:
            await http.aclose()

    app = FastAPI(title="GoldenSense Agent Gateway", version="2.0.0", lifespan=lifespan)
    allow_origins = [
        origin.strip()
        for origin in _env(
            "AGENT_ALLOW_ORIGINS",
            "http://localhost:4173,http://127.0.0.1:4173,http://localhost:8501,http://127.0.0.1:8501",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        trace_health = app.state.trace_store.health() if hasattr(app.state.trace_store, "health") else {}
        return {
            "status": "ok",
            "mode": "educational-retail-agent",
            "auth": "api-key",
            "trace_store": trace_health,
        }

    @app.get("/health/live")
    async def health_live() -> Dict[str, str]:
        return {"status": "ok", "service": "agent_gateway"}

    @app.get("/health/ready")
    async def health_ready() -> Response:
        trace_health = app.state.trace_store.health() if hasattr(app.state.trace_store, "health") else {}
        errors: List[str] = []
        if not trace_health.get("db_ready") and not trace_health.get("allow_memory_fallback"):
            errors.append("trace_store_unavailable")

        downstream: Dict[str, Any] = {}
        if getattr(app.state, "uses_injected_toolbox", False):
            downstream["toolbox"] = {"status": "skipped", "reason": "injected_toolbox"}
        else:
            http: httpx.AsyncClient = app.state.http
            health_targets = {
                "forecast": _health_url(cfg.forecast_url),
                "memory": _health_url(cfg.memory_url),
                "market": _health_url(cfg.market_snapshot_url),
                "indicators": _health_url(cfg.market_indicators_url),
                "history": _health_url(cfg.market_history_url),
                "news": _health_url(cfg.recent_news_url),
            }
            for name, url in health_targets.items():
                try:
                    resp = await http.get(url)
                    ok = resp.status_code < 400
                    downstream[name] = {
                        "status_code": resp.status_code,
                        "status": "ok" if ok else "unavailable",
                    }
                    if not ok:
                        errors.append(f"{name}_unavailable")
                except Exception as exc:
                    downstream[name] = {"status": "unavailable", "error": f"{type(exc).__name__}:{exc}"}
                    errors.append(f"{name}_unavailable")

        status = "ok" if not errors else "unavailable"
        return JSONResponse(
            status_code=200 if not errors else 503,
            content={
                "status": status,
                "trace_store": trace_health,
                "downstream": downstream,
                "errors": errors,
            },
        )

    @app.post("/api/v1/agent/analyze", response_model=AgentAnalyzeResponse)
    async def analyze(req: AgentAnalyzeRequest, request: Request) -> AgentAnalyzeResponse:
        auth_ctx = app.state.authorizer.authorize(request, internal_only=False)
        await app.state.rate_limiter.check(auth_ctx["client_id"])
        service: AgentAnalysisService = app.state.analysis_service
        return await service.analyze(req)

    @app.get("/api/v1/agent/forecasts/current", response_model=AgentForecastsResponse)
    async def current_forecasts(request: Request) -> AgentForecastsResponse:
        auth_ctx = app.state.authorizer.authorize(request, internal_only=False)
        await app.state.rate_limiter.check(auth_ctx["client_id"])
        service: AgentAnalysisService = app.state.analysis_service
        return await service.current_forecasts()

    @app.get("/api/v1/agent/dashboard/current", response_model=AgentDashboardResponse)
    async def current_dashboard(request: Request) -> AgentDashboardResponse:
        auth_ctx = app.state.authorizer.authorize(request, internal_only=False)
        await app.state.rate_limiter.check(auth_ctx["client_id"])
        service: AgentAnalysisService = app.state.analysis_service
        return await service.current_dashboard()

    @app.post("/api/v1/agent/feedback", response_model=AgentFeedbackResponse)
    async def feedback(req: AgentFeedbackRequest, request: Request) -> AgentFeedbackResponse:
        app.state.authorizer.authorize(request, internal_only=False)
        store: AgentTraceStore = app.state.trace_store
        try:
            updated = await store.save_feedback(req.analysis_id, req.rating, req.comment)
        except TraceStoreUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "trace_store_unavailable",
                    "message": f"Analysis trace store unavailable: {exc}",
                },
            ) from exc
        if not updated:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "analysis_not_found",
                    "message": f"Analysis trace not found: {req.analysis_id}",
                },
            )
        return AgentFeedbackResponse(analysis_id=req.analysis_id, status="recorded")

    @app.get("/api/v1/agent/traces/{analysis_id}", response_model=AgentTraceResponse)
    async def get_trace(analysis_id: str, request: Request) -> AgentTraceResponse:
        app.state.authorizer.authorize(request, internal_only=True)
        store: AgentTraceStore = app.state.trace_store
        try:
            payload = await store.load_trace(analysis_id)
        except TraceStoreUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "trace_store_unavailable",
                    "message": f"Analysis trace store unavailable: {exc}",
                },
            ) from exc
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "analysis_not_found",
                    "message": f"Analysis trace not found: {analysis_id}",
                },
            )
        return AgentTraceResponse(**payload)

    @app.post("/api/v1/agent/trigger", response_model=AgentTriggerResponse)
    async def trigger(req: AgentTriggerRequest, request: Request, response: Response) -> AgentTriggerResponse:
        app.state.authorizer.authorize(request, internal_only=True)
        service: AgentAnalysisService = app.state.analysis_service
        analysis_run = await service.analyze_internal(
            AgentAnalyzeRequest(
                question="请分析这条新闻对黄金的影响，并给出适合散户理解的短线建议。",
                optional_news_text=req.news_text,
                risk_profile="balanced",
                horizon="24h",
                locale="zh-CN",
            )
        )
        analysis = analysis_run.response
        bundle = analysis_run.bundle
        sentiment_score = app.state.sentiment_scorer.score(req.news_text)
        decision = _legacy_decision_from_analyze(analysis)
        rag_events = bundle.rag_events[:3]
        rag_titles = [event.headline for event in rag_events]
        quant_probability = _safe_float(bundle.forecast.get("probability"))
        xgboost_probability = _safe_float(bundle.forecast.get("xgboost_probability"))
        current_vix = float(req.manual_vix)
        if current_vix >= cfg.vix_circuit_breaker_threshold:
            decision = AgentDecision(
                action="HOLD",
                confidence=0.2,
                horizon="T+1",
                reasoning_summary=analysis.summary_card.reasons[0],
                risk_warning="Manual VIX override triggered conservative hold.",
            )
            risk_result = {
                "decision": "REJECTED",
                "executed_position": 0.0,
                "current_vix": current_vix,
                "vix_threshold": cfg.vix_circuit_breaker_threshold,
                "notes": f"Manual VIX override triggered hold at VIX={current_vix}",
            }
        else:
            risk_result = _legacy_risk_result(analysis, current_vix, cfg.vix_circuit_breaker_threshold)
        impact = compute_impact_breakdown(
            emotion_weight=sentiment_score,
            rag_events=rag_events,
            decision=decision,
            xgboost_probability=xgboost_probability,
            risk_result=risk_result,
        )
        response.headers["X-Impact-Breakdown"] = json.dumps(impact.model_dump(), ensure_ascii=True, separators=(",", ":"))
        response.headers["X-Trace-Analysis-Id"] = analysis.analysis_id
        return AgentTriggerResponse(
            decision=decision,
            finbert_sentiment_score=sentiment_score,
            rag_top_3_event_titles=rag_titles,
            rag_top_3_events=rag_events,
            xgboost_probability=xgboost_probability,
            quant_probability=quant_probability,
            risk_result=risk_result,
            impact_breakdown=impact,
            timing_ms=analysis.timing_ms,
        )

    return app


app = create_app()
