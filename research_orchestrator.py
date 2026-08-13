"""Deterministic expert orchestration over a gated ResearchCase blackboard."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from evidence_shield import EvidencePacket
from market_view import build_market_view
from research_case import (
    AgentView,
    AuditReport,
    HorizonScenario,
    HorizonStrategy,
    ModelCard,
    OutcomeCheckpoint,
    ResearchCase,
    ResearchConflict,
)

Expert = Literal[
    "macro_event", "technical_flows", "long_term_fundamental", "quant_model_risk"
]

_MACRO = ("cpi", "inflation", "fed", "fomc", "rate", "yield", "usd", "dollar", "美联储", "通胀", "利率")
_FLOW = ("etf", "cftc", "flow", "trend", "momentum", "technical", "price", "资金流", "趋势", "技术")
_LONG = ("central bank", "reserve", "mine", "supply", "de-dollar", "structural", "央行", "储备", "矿产", "供给")


def select_experts(question: str, evidence_text: str) -> List[Expert]:
    combined = f"{question} {evidence_text}".lower()
    selected: List[Expert] = []
    if any(term in combined for term in _MACRO):
        selected.append("macro_event")
    if any(term in combined for term in _FLOW):
        selected.append("technical_flows")
    if any(term in combined for term in _LONG):
        selected.append("long_term_fundamental")
    selected.append("quant_model_risk")
    return selected


def _stance(text: str, positive: tuple[str, ...], negative: tuple[str, ...]) -> str:
    lowered = text.lower()
    pos = sum(term in lowered for term in positive)
    neg = sum(term in lowered for term in negative)
    if pos > neg:
        return "bullish"
    if neg > pos:
        return "bearish"
    return "neutral"


def _expert_view(expert: Expert, text: str, fact_ids: List[str], ctx: Dict[str, Any]) -> AgentView:
    if expert == "macro_event":
        stance = _stance(
            text,
            ("rate cut", "cuts", "below consensus", "real yields fell", "dollar weakened", "降息", "低于预期"),
            ("rate hike", "above consensus", "real yields rose", "dollar strengthened", "加息", "高于预期"),
        )
        return AgentView(
            agent=expert, horizon="mid_term", stance=stance, confidence=0.68,
            thesis="Macro transmission is assessed through surprise, real yields and the dollar; evidence is not treated as a trading instruction.",
            supporting_fact_ids=fact_ids, counter_fact_ids=[],
            invalidation=["The realised real-yield or dollar response reverses the stated transmission."],
        )
    if expert == "technical_flows":
        stance = _stance(
            text,
            ("etf inflow", "inflows", "uptrend", "breakout", "流入", "向上突破"),
            ("etf outflow", "outflows", "pressure gold", "downtrend", "流出", "下行趋势"),
        )
        return AgentView(
            agent=expert, horizon="short_term", stance=stance, confidence=0.58,
            thesis="Price structure and positioning are confirmation evidence, with special attention to whether the event is already priced.",
            supporting_fact_ids=fact_ids, counter_fact_ids=[],
            invalidation=["ETF/CFTC flow or trend confirmation changes sign."],
        )
    if expert == "long_term_fundamental":
        stance = _stance(
            text,
            ("central bank demand", "reserve diversification", "constrained supply", "央行购金", "储备多元化"),
            ("mine supply increased", "reserve sales", "央行售金", "供给大增"),
        )
        return AgentView(
            agent=expert, horizon="long_term", stance=stance, confidence=0.55,
            thesis="Structural demand, reserve policy and mine supply are separated from near-term event noise.",
            supporting_fact_ids=fact_ids, counter_fact_ids=[],
            invalidation=["Reserve behaviour or structural supply assumptions reverse."],
        )

    composite = float((ctx.get("macro_factors") or {}).get("composite", 0.5))
    quant_stance = "bullish" if composite >= 0.55 else "bearish" if composite <= 0.45 else "neutral"
    degradation = list((ctx.get("degraded") or {}).keys())
    return AgentView(
        agent="quant_model_risk", horizon="mid_term", stance=quant_stance,
        confidence=0.72 if not degradation else 0.4,
        thesis="Validated production models provide distribution, regime and valuation context; unvalidated challengers are observation-only.",
        supporting_fact_ids=[], counter_fact_ids=[],
        invalidation=["Walk-forward coverage, calibration or data-freshness controls fail."],
        degradation_flags=degradation,
    )


def _model_registry(ctx: Dict[str, Any]) -> List[ModelCard]:
    flagship = ctx.get("flagship") or {}
    metrics = flagship.get("metrics") or {}
    sample = flagship.get("sample") or {}
    governance = flagship.get("governance") or {}
    flagship_state = "production" if governance.get("mode") == "champion" and metrics else "degraded"
    cards = [
        ModelCard(model_id="har_rv", label="HAR-RV volatility", state="production", role="champion", oos_metric="coverage diagnostics", can_influence_strategy=True),
        ModelCard(model_id="hmm", label="HMM regime", state="production", role="champion", oos_metric="walk-forward posterior", can_influence_strategy=True),
        ModelCard(model_id="macro_factors", label="Macro factor composite", state="production", role="champion", oos_metric="causal factor rules", can_influence_strategy=True),
        ModelCard(
            model_id="flagship", label="Integrated flagship", state=flagship_state, role="champion",
            oos_metric=(f"Sharpe {metrics.get('sharpe')} / max drawdown {float(metrics.get('max_drawdown', 0)):.0%}" if metrics else None),
            sample_period=(f"{sample.get('start')} → {sample.get('end')}" if sample else None),
            degradation_reason=None if flagship_state == "production" else "governance_not_champion",
            can_influence_strategy=flagship_state == "production",
        ),
    ]
    for model_id, label in (
        ("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("gru", "GRU"),
        ("lstm", "LSTM"), ("transformer", "Transformer"),
    ):
        cards.append(ModelCard(
            model_id=model_id, label=label, state="watch", role="challenger",
            degradation_reason="not_walk_forward_validated", can_influence_strategy=False,
        ))
    return cards


def _scenario(label: str, probability: float, description: str) -> HorizonScenario:
    return HorizonScenario(label=label, probability=probability, description=description)


def _blocked_strategy(horizon: str, now: datetime) -> HorizonStrategy:
    return HorizonStrategy(
        horizon=horizon, stance="abstain", confidence=0.0, priced_in="uncertain",
        base=_scenario("base", 0.6, "Evidence is blocked; no research conclusion is published."),
        upside=_scenario("upside", 0.2, "Unavailable until trusted evidence is supplied."),
        downside=_scenario("downside", 0.2, "Unavailable until trusted evidence is supplied."),
        triggers=["Supply clean, traceable evidence."],
        invalidation=["Blocked evidence cannot validate or invalidate a market view."],
        next_review_at=now + timedelta(days=1), degradation_flags=["evidence_gate_blocked"],
    )


def _strategies(ctx: Dict[str, Any], *, blocked: bool, now: datetime) -> Dict[str, HorizonStrategy]:
    if blocked:
        return {h: _blocked_strategy(h, now) for h in ("short_term", "mid_term", "long_term")}
    book = build_market_view(ctx)
    composite = float((ctx.get("macro_factors") or {}).get("composite", 0.5))
    mid_stance = "bullish" if composite >= 0.55 else "bearish" if composite <= 0.45 else "neutral"
    fair = ctx.get("fair_value") or {}
    long_stance = "risk" if fair.get("regime_break") else "neutral"
    short = book.get("short_term") or {}
    mid = book.get("mid_term") or {}
    long = book.get("long_term") or {}
    return {
        "short_term": HorizonStrategy(
            horizon="short_term", stance="risk" if short.get("available") else "abstain",
            confidence=0.45 if short.get("available") else 0.0, priced_in="uncertain",
            base=_scenario("base", 0.6, short.get("core_view", "Short-term distribution unavailable.")),
            upside=_scenario("upside", 0.2, "Gold holds the upper volatility band when yields and USD confirm."),
            downside=_scenario("downside", 0.2, "Gold tests the lower band if yields/USD oppose the event narrative."),
            triggers=["HAR-RV band coverage", "real-yield and USD reaction", "ETF/CFTC confirmation"],
            invalidation=list(short.get("invalidation") or ["Volatility model coverage fails."]),
            next_review_at=now + timedelta(days=7),
            degradation_flags=["unsupported_direction_model_direction_abstained"],
        ),
        "mid_term": HorizonStrategy(
            horizon="mid_term", stance=mid_stance, confidence=0.65, priced_in="uncertain",
            base=_scenario("base", 0.6, mid.get("core_view", "Mid-term state remains uncertain.")),
            upside=_scenario("upside", 0.2, "Real yields and USD turn into a sustained macro tailwind."),
            downside=_scenario("downside", 0.2, "Tighter policy and USD strength dominate flows."),
            triggers=["HMM state probability", "macro composite crossing 0.5", "flow confirmation"],
            invalidation=list(mid.get("invalidation") or ["Regime probability loses dominance."]),
            next_review_at=now + timedelta(days=30),
        ),
        "long_term": HorizonStrategy(
            horizon="long_term", stance=long_stance, confidence=0.55, priced_in="uncertain",
            base=_scenario("base", 0.6, long.get("core_view", "Long-term valuation anchor unavailable.")),
            upside=_scenario("upside", 0.2, "Reserve diversification sustains a structural premium."),
            downside=_scenario("downside", 0.2, "Real-rate anchor and reserve demand normalise."),
            triggers=["fair-value z-score", "central-bank demand", "structural break diagnostics"],
            invalidation=list(long.get("invalidation") or ["Valuation relationship breaks."]),
            next_review_at=now + timedelta(days=90),
        ),
    }


def _conflicts(views: List[AgentView]) -> List[ResearchConflict]:
    directional = {view.stance for view in views if view.stance in {"bullish", "bearish"}}
    if directional != {"bullish", "bearish"}:
        return []
    bulls = [view.agent for view in views if view.stance == "bullish"]
    bears = [view.agent for view in views if view.stance == "bearish"]
    return [ResearchConflict(
        topic="cross-horizon evidence disagreement",
        majority_view="No majority is promoted without horizon and evidence weighting.",
        minority_view=f"Bullish: {', '.join(bulls)}; bearish: {', '.join(bears)}",
        agent_ids=bulls + bears,
        resolution="Preserve the minority view and expose separate horizon triggers; do not majority-vote it away.",
    )]


def build_research_case(
    question: str,
    evidence: EvidencePacket,
    quant_ctx: Dict[str, Any],
    *,
    mode: Literal["draft", "full"] = "full",
    now: Optional[datetime] = None,
) -> ResearchCase:
    del mode  # both modes share deterministic facts; narration is a later optional layer
    now = now or datetime.now(timezone.utc)
    blocked = any(gate.decision == "block" for gate in evidence.gates)
    accepted = evidence.accepted_facts
    text = " ".join(fact.text for fact in accepted)
    experts = select_experts(question, text)
    views = [] if blocked else [
        _expert_view(expert, text, [fact.claim_id for fact in accepted], quant_ctx)
        for expert in experts
    ]
    strategies = _strategies(quant_ctx, blocked=blocked, now=now)
    if not blocked:
        views.append(AgentView(
            agent="strategy_arbitrator", horizon="mid_term",
            stance=strategies["mid_term"].stance,
            confidence=strategies["mid_term"].confidence,
            thesis="Views are combined by horizon, evidence quality, freshness and model governance; disagreement remains visible.",
            supporting_fact_ids=[fact.claim_id for fact in accepted],
            counter_fact_ids=[], invalidation=strategies["mid_term"].invalidation,
        ))
    stale = bool(quant_ctx.get("data_stale"))
    status = "blocked" if blocked else "degraded" if stale or not accepted else "complete"
    issues = [gate.reason for gate in evidence.gates if gate.decision in {"block", "abstain"}]
    if stale:
        issues.append("quantitative context is stale")
    case_seed = f"{question}|{evidence.document.sha256}|{now.isoformat()}"
    case_id = "rc_" + hashlib.sha256(case_seed.encode("utf-8")).hexdigest()[:18]
    quant_keys = (
        "data_source", "data_asof", "data_age_days", "data_stale", "vol_bands",
        "regime_posterior", "macro_factors", "fair_value", "scenario_cone",
        "flagship", "degraded",
    )
    quant_pack = {key: quant_ctx.get(key) for key in quant_keys if key in quant_ctx}
    return ResearchCase(
        case_id=case_id, question=question, created_at=now,
        data_asof=quant_ctx.get("data_asof"), status=status,
        evidence_documents=[evidence.document], fact_claims=evidence.facts,
        gate_report=evidence.gates,
        market_snapshot={
            key: quant_ctx.get(key) for key in ("latest_price", "data_asof", "data_stale")
            if key in quant_ctx
        },
        quant_pack=quant_pack, model_registry=_model_registry(quant_ctx),
        agent_views=views, conflicts=_conflicts(views), horizon_strategy=strategies,
        audit_report=AuditReport(passed=not blocked, issues=issues, checked_at=now),
        outcome_schedule=[
            OutcomeCheckpoint(horizon="short_term", due_at=now + timedelta(days=21)),
            OutcomeCheckpoint(horizon="mid_term", due_at=now + timedelta(days=180)),
            OutcomeCheckpoint(horizon="long_term", due_at=now + timedelta(days=365)),
        ],
    )
