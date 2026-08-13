"""Shared, auditable research-case domain models.

The case is the blackboard shared by ingestion, expert research, strategy,
personalisation and forward review.  Raw uploaded bytes are intentionally not
part of this model; persistence stores only hashes, located claims and results.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from threading import RLock
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

GateStatus = Literal["pass", "review", "block", "abstain"]
FactStatus = Literal["accepted", "review", "blocked"]
CaseStatus = Literal["created", "gated", "researching", "complete", "blocked", "degraded"]
Horizon = Literal["short_term", "mid_term", "long_term"]
Stance = Literal["bullish", "bearish", "neutral", "risk", "abstain"]
ModelState = Literal["production", "watch", "degraded", "retired"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceDocument(StrictModel):
    document_id: str
    kind: Literal["url", "pdf", "image", "question", "api"]
    filename: Optional[str] = None
    sha256: str = Field(min_length=64, max_length=64)
    source_url: Optional[str] = None
    content_type: str
    source_tier: Literal["primary", "secondary", "unknown"] = "unknown"
    retrieved_at: datetime
    published_at: Optional[datetime] = None
    persisted_raw: Literal[False] = False
    extraction_status: Literal["complete", "partial", "abstain"] = "complete"
    degradation_flags: List[str] = Field(default_factory=list)


class FactClaim(StrictModel):
    claim_id: str
    document_id: str
    text: str = Field(min_length=1, max_length=4000)
    locator: str = Field(min_length=1, max_length=300)
    status: FactStatus
    confidence: float = Field(ge=0.0, le=1.0)
    units: Optional[str] = None
    published_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)


class GateDecision(StrictModel):
    gate: Literal[
        "access",
        "provenance_time",
        "fact_location",
        "ai_attack",
        "dedup_replay",
        "consistency",
        "market_coherence",
        "output_audit",
    ]
    decision: GateStatus
    reason: str
    confidence_multiplier: float = Field(ge=0.0, le=1.0)
    evidence_refs: List[str] = Field(default_factory=list)


class AgentView(StrictModel):
    agent: Literal[
        "macro_event",
        "technical_flows",
        "long_term_fundamental",
        "quant_model_risk",
        "strategy_arbitrator",
    ]
    horizon: Horizon
    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    supporting_fact_ids: List[str] = Field(default_factory=list)
    counter_fact_ids: List[str] = Field(default_factory=list)
    invalidation: List[str] = Field(default_factory=list)
    degradation_flags: List[str] = Field(default_factory=list)


class ResearchConflict(StrictModel):
    topic: str
    majority_view: str
    minority_view: str
    agent_ids: List[str]
    resolution: str


class HorizonScenario(StrictModel):
    label: Literal["base", "upside", "downside"]
    probability: float = Field(ge=0.0, le=1.0)
    description: str


class HorizonStrategy(StrictModel):
    horizon: Horizon
    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    priced_in: Literal["yes", "partial", "no", "uncertain"]
    base: HorizonScenario
    upside: HorizonScenario
    downside: HorizonScenario
    triggers: List[str]
    invalidation: List[str]
    next_review_at: datetime
    degradation_flags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> "HorizonStrategy":
        total = self.base.probability + self.upside.probability + self.downside.probability
        if abs(total - 1.0) > 1e-6:
            raise ValueError("scenario probabilities must sum to 1")
        return self


class ModelCard(StrictModel):
    model_id: str
    label: str
    state: ModelState
    role: Literal["champion", "challenger"]
    oos_metric: Optional[str] = None
    sample_period: Optional[str] = None
    degradation_reason: Optional[str] = None
    can_influence_strategy: bool = False


class AuditReport(StrictModel):
    passed: bool
    issues: List[str]
    checked_at: Optional[datetime] = None


class OutcomeCheckpoint(StrictModel):
    horizon: Horizon
    due_at: datetime
    status: Literal["scheduled", "matured", "scored"] = "scheduled"


class ResearchCase(StrictModel):
    case_id: str
    question: str = Field(min_length=1, max_length=2000)
    asset: Literal["XAUUSD"] = "XAUUSD"
    created_at: datetime
    data_asof: Optional[str] = None
    status: CaseStatus
    investor_profile: Optional[Dict[str, Any]] = None
    evidence_documents: List[EvidenceDocument] = Field(default_factory=list)
    fact_claims: List[FactClaim] = Field(default_factory=list)
    gate_report: List[GateDecision] = Field(default_factory=list)
    market_snapshot: Dict[str, Any] = Field(default_factory=dict)
    quant_pack: Dict[str, Any] = Field(default_factory=dict)
    model_registry: List[ModelCard] = Field(default_factory=list)
    agent_views: List[AgentView] = Field(default_factory=list)
    conflicts: List[ResearchConflict] = Field(default_factory=list)
    horizon_strategy: Dict[str, HorizonStrategy] = Field(default_factory=dict)
    personalized_brief: Optional[Dict[str, Any]] = None
    audit_report: AuditReport
    outcome_schedule: List[OutcomeCheckpoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def agent_references_only_accepted_facts(self) -> "ResearchCase":
        accepted = {claim.claim_id for claim in self.fact_claims if claim.status == "accepted"}
        for view in self.agent_views:
            refs = set(view.supporting_fact_ids + view.counter_fact_ids)
            unknown = refs - accepted
            if unknown:
                raise ValueError(
                    "agent evidence must reference an accepted fact: " + ", ".join(sorted(unknown))
                )
        return self


class ResearchCaseStore:
    """Small bounded store; production can inject a durable implementation."""

    def __init__(self, *, max_items: int = 200) -> None:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        self.max_items = max_items
        self._items: "OrderedDict[str, str]" = OrderedDict()
        self._lock = RLock()

    def save(self, case: ResearchCase) -> ResearchCase:
        payload = case.model_dump_json()
        with self._lock:
            self._items.pop(case.case_id, None)
            self._items[case.case_id] = payload
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return ResearchCase.model_validate_json(payload)

    def get(self, case_id: str) -> Optional[ResearchCase]:
        with self._lock:
            payload = self._items.get(case_id)
            if payload is None:
                return None
            self._items.move_to_end(case_id)
        return ResearchCase.model_validate_json(payload)

    def list_recent(self, *, limit: int = 20) -> List[ResearchCase]:
        with self._lock:
            payloads = list(self._items.values())[-max(0, limit):]
        return [ResearchCase.model_validate_json(payload) for payload in reversed(payloads)]
