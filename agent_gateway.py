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
from pydantic import BaseModel, ConfigDict, Field

from service_contracts import MarketSnapshotResponse, NewsEventItem, RecentNewsResponse

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


class AgentAnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=3000)
    optional_news_text: Optional[str] = Field(default=None, max_length=5000)
    risk_profile: RiskProfile = "conservative"
    horizon: PublicHorizon = "24h"
    locale: Literal["zh-CN"] = "zh-CN"


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


class HorizonForecastCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    horizon: PublicHorizon
    stance: SummaryStance
    confidence_band: ConfidenceBand
    action: SummaryAction
    probability: float = Field(ge=0.0, le=1.0)
    basis: ForecastBasis
    reasons: List[str] = Field(min_length=2, max_length=4)


AgentAnalyzeResponse.model_rebuild()


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


def _forecast_is_degraded(payload: Dict[str, Any]) -> bool:
    return str(payload.get("service_status", "ok")) != "ok"


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
    return any(item.source == "synthetic_fallback" for item in snapshot.instruments)


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
        snapshot, forecast_map, news, memory_lookup, tool_trace = await self._gather(req, evidence_query=evidence_query)
        rag_events = memory_lookup.items
        news_sentiment = self._derive_news_sentiment(req, news)
        risk_profile = self._toolbox.get_user_risk_profile(req.risk_profile)
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
            self._build_horizon_forecast_card(
                horizon=horizon,
                forecast=forecast_map[horizon],
                snapshot=snapshot,
                news=news,
                news_sentiment=news_sentiment,
                rag_events=rag_events,
                memory_lookup=memory_lookup,
                macro_context=macro_context,
                risk_profile=risk_profile,
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
                    },
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

    async def _gather(
        self, req: AgentAnalyzeRequest, *, evidence_query: str
    ) -> Tuple[MarketSnapshotResponse, Dict[PublicHorizon, Dict[str, Any]], RecentNewsResponse, HistoricalEventsLookup, List[Dict[str, Any]]]:
        try:
            timed_snapshot, timed_forecast_24h, timed_forecast_7d, timed_forecast_30d, timed_news, timed_rag = await asyncio.gather(
                self._timed_tool("get_market_snapshot", self._toolbox.get_market_snapshot()),
                self._timed_optional_tool(
                    "get_quant_forecast_24h",
                    self._toolbox.get_quant_forecast("24h"),
                    lambda exc: _fallback_quant_forecast(f"{type(exc).__name__}:{exc}"),
                ),
                self._timed_optional_tool(
                    "get_quant_forecast_7d",
                    self._toolbox.get_quant_forecast("7d"),
                    lambda exc: _fallback_quant_forecast(f"{type(exc).__name__}:{exc}"),
                ),
                self._timed_optional_tool(
                    "get_quant_forecast_30d",
                    self._toolbox.get_quant_forecast("30d"),
                    lambda exc: _fallback_quant_forecast(f"{type(exc).__name__}:{exc}"),
                ),
                self._timed_optional_tool(
                    "search_recent_news",
                    self._toolbox.search_recent_news(evidence_query, limit=6),
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
        tool_trace = [
            timed_snapshot[1],
            timed_forecast_24h[1],
            timed_forecast_7d[1],
            timed_forecast_30d[1],
            timed_news[1],
            timed_rag[1],
        ]
        snapshot, news, memory_lookup = timed_snapshot[0], timed_news[0], _normalize_memory_lookup(timed_rag[0])
        forecast_map: Dict[PublicHorizon, Dict[str, Any]] = {
            "24h": timed_forecast_24h[0],
            "7d": timed_forecast_7d[0],
            "30d": timed_forecast_30d[0],
        }
        return snapshot, forecast_map, news, memory_lookup, tool_trace

    def _tool_trace_entry(self, name: str, payload: Any, *, elapsed_ms: int, status: str, error: Optional[str] = None) -> Dict[str, Any]:
        source_status = "ok"
        degraded = False
        fallback_reason = None
        source_freshness_seconds = None

        if isinstance(payload, RecentNewsResponse):
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
            or memory_lookup.status != "ok"
        )
        basis = _forecast_basis(forecast)
        low_confidence_threshold = {"24h": 0.56, "7d": 0.58, "30d": 0.6}[horizon]
        is_low_confidence = probability < low_confidence_threshold
        is_high_risk = bool(
            snapshot.is_stale
            or has_degraded_inputs
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
                    excerpt=f"历史记忆检索当前处于 {bundle.memory_status} 状态：{bundle.memory_degraded_reason or '无可用原因'}。",
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
                excerpt=str(bundle.risk_profile["description"]),
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
                    f"历史记忆检索当前处于 {bundle.memory_status} 状态，原因：{bundle.memory_degraded_reason or '未返回'}。"
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
                takeaway=f"{bundle.risk_profile['label']}：{bundle.risk_profile['description']}",
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
        if bundle.is_high_risk or bundle.has_conflict or bundle.is_low_confidence:
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
        invalidators = [
            "如果美元和实际利率同步快速走强，当前观点需要重新评估。",
            "如果接下来 1-2 个交易时段新闻方向反转，历史类比可能失效。",
            "如果 VIX 升到 30 以上或数据明显陈旧，应立即降级为观望。",
        ]
        disclaimer = "本内容仅用于帮助理解黄金市场，不构成个性化投资建议，也不替代你自己的风险决策。"

        if bundle.has_degraded_inputs:
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


def create_app(
    *,
    toolbox: Optional[HttpResearchToolbox] = None,
    narrator: Optional[OpenAINarrator] = None,
    sentiment_scorer: Optional[BaseSentimentScorer] = None,
    trace_store: Optional[AgentTraceStore] = None,
    http_client: Optional[httpx.AsyncClient] = None,
) -> FastAPI:
    tool_timeout_seconds = float(_env("AGENT_TOOL_TIMEOUT_SECONDS", "4.0"))
    tool_connect_timeout_seconds = float(_env("AGENT_TOOL_CONNECT_TIMEOUT_SECONDS", "1.5"))
    app_env = _env("APP_ENV", "development").lower()
    cfg = AgentGatewayConfig(
        forecast_url=_env("FORECAST_URL", "http://localhost:8010/api/v1/forecast"),
        memory_url=_env("MEMORY_URL", "http://localhost:8012/api/v1/memory/search"),
        market_snapshot_url=_env("MARKET_SNAPSHOT_URL", "http://localhost:8014/api/v1/market/snapshot/latest"),
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
    if app_env == "production" and (not public_keys_raw or not internal_keys_raw):
        raise RuntimeError("AGENT_PUBLIC_API_KEYS and AGENT_INTERNAL_API_KEYS must be set in production.")
    public_api_keys = _split_csv(public_keys_raw or "dev-public-key")
    internal_api_keys = _split_csv(internal_keys_raw or "dev-internal-key")
    analyze_rate_limit_per_minute = int(_env("AGENT_ANALYZE_RATE_LIMIT_PER_MINUTE", "60"))
    analyze_rate_limit_window_seconds = int(_env("AGENT_ANALYZE_RATE_LIMIT_WINDOW_SECONDS", "60"))
    allow_trace_memory_fallback = (
        os.environ.get("AGENT_ALLOW_TRACE_MEMORY_FALLBACK", "1" if app_env != "production" else "0") != "0"
    )
    trace_memory_ttl_seconds = int(_env("AGENT_TRACE_MEMORY_TTL_SECONDS", "3600"))
    trace_memory_max_items = int(_env("AGENT_TRACE_MEMORY_MAX_ITEMS", "200"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        own_http = http_client is None
        http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(tool_timeout_seconds, connect=tool_connect_timeout_seconds)
        )
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
        return {
            "status": "ok",
            "mode": "educational-retail-agent",
            "auth": "api-key",
        }

    @app.post("/api/v1/agent/analyze", response_model=AgentAnalyzeResponse)
    async def analyze(req: AgentAnalyzeRequest, request: Request) -> AgentAnalyzeResponse:
        auth_ctx = app.state.authorizer.authorize(request, internal_only=False)
        await app.state.rate_limiter.check(auth_ctx["client_id"])
        service: AgentAnalysisService = app.state.analysis_service
        return await service.analyze(req)

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
