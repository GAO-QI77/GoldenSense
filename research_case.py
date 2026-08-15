"""Shared, auditable research-case domain models.

The case is the blackboard shared by ingestion, expert research, strategy,
personalisation and forward review.  Raw uploaded bytes are intentionally not
part of this model; persistence stores only hashes, located claims and results.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

GateStatus = Literal["pass", "review", "block", "abstain"]
FactStatus = Literal["accepted", "review", "blocked"]
CaseStatus = Literal["created", "gated", "researching", "complete", "blocked", "degraded"]
Horizon = Literal["short_term", "mid_term", "long_term"]
Stance = Literal["bullish", "bearish", "neutral", "risk", "abstain"]
ModelState = Literal["production", "watch", "degraded", "retired"]
EvidenceDomain = Literal[
    "macro_event", "technical_flows", "long_term_fundamental", "product_rules", "general"
]
ProbabilityKind = Literal["calibrated_probability", "research_weight", "abstention"]


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
    domains: List[EvidenceDomain] = Field(default_factory=lambda: ["general"])


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


class ConfidenceBasis(StrictModel):
    method: str
    calibrated: bool = False
    sample_size: Optional[int] = Field(default=None, ge=1)
    calibration_period: Optional[str] = None
    source_refs: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def calibrated_confidence_has_audit_metadata(self) -> "ConfidenceBasis":
        if self.calibrated and (
            self.sample_size is None
            or not self.calibration_period
            or not self.source_refs
        ):
            raise ValueError(
                "calibrated confidence requires sample size, calibration period and sources"
            )
        return self


def _unspecified_confidence_basis() -> ConfidenceBasis:
    return ConfidenceBasis(
        method="legacy_unspecified",
        calibrated=False,
        source_refs=[],
    )


class AgentView(StrictModel):
    agent: Literal[
        "macro_event",
        "technical_flows",
        "long_term_fundamental",
        "product_rules_review",
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
    confidence_basis: ConfidenceBasis = Field(default_factory=_unspecified_confidence_basis)


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
    probability_kind: ProbabilityKind = "research_weight"
    method: str = "legacy_unspecified"
    sample_size: Optional[int] = Field(default=None, ge=1)
    calibration_error: Optional[float] = Field(default=None, ge=0.0, le=1.0)


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
    confidence_basis: ConfidenceBasis = Field(default_factory=_unspecified_confidence_basis)
    priced_in_basis: str = "not_assessed"

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> "HorizonStrategy":
        total = self.base.probability + self.upside.probability + self.downside.probability
        if abs(total - 1.0) > 1e-6:
            raise ValueError("scenario probabilities must sum to 1")
        scenarios = (self.base, self.upside, self.downside)
        probability_kinds = {scenario.probability_kind for scenario in scenarios}
        if len(probability_kinds) != 1:
            raise ValueError("scenarios must use the same probability kind")
        if probability_kinds == {"calibrated_probability"} and any(
            scenario.sample_size is None
            or scenario.calibration_error is None
            or scenario.method == "legacy_unspecified"
            for scenario in scenarios
        ):
            raise ValueError(
                "calibrated probability requires sample size, calibration error and method"
            )
        if probability_kinds == {"abstention"} and self.stance != "abstain":
            raise ValueError("abstention probabilities require an abstain strategy")
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
    entry_price: Optional[float] = Field(default=None, gt=0.0)
    realized_price: Optional[float] = Field(default=None, gt=0.0)
    realized_return: Optional[float] = None
    direction_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    neutral_band_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scenario_outcome: Optional[Literal["base", "upside", "downside"]] = None
    scenario_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scenario_score_kind: Optional[Literal["brier", "weight_brier"]] = None
    confidence_error: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scoring_method: Optional[str] = None
    scored_at: Optional[datetime] = None


class ResearchNarrative(StrictModel):
    """Optional presentation layer over the deterministic research record.

    The structured case remains authoritative.  In ``full`` mode an LLM may
    rewrite these fields, but the gateway re-checks numbers and directive
    language before persisting the result.
    """

    overview: str = Field(min_length=1, max_length=4000)
    horizon_notes: Dict[Horizon, str] = Field(default_factory=dict)
    watchlist: List[str] = Field(default_factory=list)
    generated_by: Literal["deterministic_draft", "llm"] = "deterministic_draft"
    degradation_flags: List[str] = Field(default_factory=list)
    critic_report: Optional[Dict[str, Any]] = None


class ResearchCase(StrictModel):
    case_id: str
    owner_hash: Optional[str] = Field(default=None, min_length=16, max_length=64)
    question: str = Field(min_length=1, max_length=2000)
    asset: Literal["XAUUSD"] = "XAUUSD"
    created_at: datetime
    data_asof: Optional[str] = None
    status: CaseStatus
    research_mode: Literal["draft", "full"] = "full"
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
    narrative: Optional[ResearchNarrative] = None
    personalized_brief: Optional[Dict[str, Any]] = None
    audit_report: AuditReport
    outcome_schedule: List[OutcomeCheckpoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def agent_references_only_accepted_facts(self) -> "ResearchCase":
        accepted = {
            claim.claim_id: claim
            for claim in self.fact_claims
            if claim.status == "accepted"
        }
        agent_domains = {
            "macro_event": "macro_event",
            "technical_flows": "technical_flows",
            "long_term_fundamental": "long_term_fundamental",
        }
        for view in self.agent_views:
            refs = set(view.supporting_fact_ids + view.counter_fact_ids)
            unknown = refs - set(accepted)
            if unknown:
                raise ValueError(
                    "agent evidence must reference an accepted fact: " + ", ".join(sorted(unknown))
                )
            expected_domain = agent_domains.get(view.agent)
            if expected_domain:
                cross_domain = sorted(
                    ref
                    for ref in refs
                    if expected_domain not in accepted[ref].domains
                )
                if cross_domain:
                    raise ValueError(
                        "domain agent must reference its own evidence domain: "
                        + ", ".join(cross_domain)
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
        # Defense in depth: investor inputs and their derived brief are
        # request-scoped, even if a caller accidentally attaches them.
        durable_case = case.model_copy(update={
            "investor_profile": None,
            "personalized_brief": None,
        })
        payload = durable_case.model_dump_json()
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

    def list_all_latest(self) -> List[ResearchCase]:
        return self.list_recent(limit=self.max_items)


class JsonlResearchCaseStore(ResearchCaseStore):
    """Append-only case ledger with a bounded in-memory latest-value index."""

    def __init__(self, path: str | Path, *, max_items: int = 200) -> None:
        super().__init__(max_items=max_items)
        self.path = Path(path)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    case = ResearchCase.model_validate_json(line)
                except Exception:
                    continue
                ResearchCaseStore.save(self, case)

    def save(self, case: ResearchCase) -> ResearchCase:
        durable_case = case.model_copy(update={
            "investor_profile": None,
            "personalized_brief": None,
        })
        payload = durable_case.model_dump_json()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()
            self._items.pop(case.case_id, None)
            self._items[case.case_id] = payload
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return ResearchCase.model_validate_json(payload)

    def list_all_latest(self) -> List[ResearchCase]:
        latest: "OrderedDict[str, ResearchCase]" = OrderedDict()
        with self._lock:
            if not self.path.exists():
                return []
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    case = ResearchCase.model_validate_json(line)
                except Exception:
                    continue
                latest.pop(case.case_id, None)
                latest[case.case_id] = case
        return list(latest.values())


def score_due_outcomes(
    case: ResearchCase,
    *,
    as_of: datetime,
    price_at: Callable[[datetime], Optional[float]],
) -> ResearchCase:
    """Score each matured checkpoint at its own due-date market close."""
    neutral_bands = {"short_term": 0.015, "mid_term": 0.04, "long_term": 0.08}
    checkpoints: List[OutcomeCheckpoint] = []
    for checkpoint in case.outcome_schedule:
        if checkpoint.status in {"scheduled", "matured"} and checkpoint.due_at <= as_of:
            realized_price = price_at(checkpoint.due_at)
            if checkpoint.entry_price is None or realized_price is None or realized_price <= 0:
                checkpoints.append(checkpoint.model_copy(update={"status": "matured"}))
                continue
            realized_return = realized_price / checkpoint.entry_price - 1.0
            strategy = case.horizon_strategy.get(checkpoint.horizon)
            score: Optional[float] = None
            neutral_band = neutral_bands[checkpoint.horizon]
            scenario_outcome = (
                "upside" if realized_return > neutral_band
                else "downside" if realized_return < -neutral_band
                else "base"
            )
            scenario_score: Optional[float] = None
            scenario_score_kind: Optional[str] = None
            confidence_error: Optional[float] = None
            if strategy and strategy.stance == "bullish":
                score = 1.0 if realized_return > 0 else 0.0
            elif strategy and strategy.stance == "bearish":
                score = 1.0 if realized_return < 0 else 0.0
            elif strategy and strategy.stance == "neutral":
                score = 1.0 if abs(realized_return) <= neutral_band else 0.0
            if strategy:
                forecasts = {
                    "base": strategy.base.probability,
                    "upside": strategy.upside.probability,
                    "downside": strategy.downside.probability,
                }
                kinds = {
                    strategy.base.probability_kind,
                    strategy.upside.probability_kind,
                    strategy.downside.probability_kind,
                }
                if kinds != {"abstention"}:
                    scenario_score = sum(
                        (forecast - (1.0 if label == scenario_outcome else 0.0)) ** 2
                        for label, forecast in forecasts.items()
                    ) / 3.0
                    scenario_score_kind = (
                        "brier" if kinds == {"calibrated_probability"} else "weight_brier"
                    )
                if score is not None:
                    confidence_error = abs(strategy.confidence - score)
            checkpoints.append(checkpoint.model_copy(update={
                "status": "scored",
                "realized_price": realized_price,
                "realized_return": realized_return,
                "direction_score": score,
                "neutral_band_pct": neutral_band,
                "scenario_outcome": scenario_outcome,
                "scenario_score": scenario_score,
                "scenario_score_kind": scenario_score_kind,
                "confidence_error": confidence_error,
                "scoring_method": "horizon_band_multiclass_v1",
                "scored_at": as_of,
            }))
        else:
            checkpoints.append(checkpoint)
    return case.model_copy(update={"outcome_schedule": checkpoints})
