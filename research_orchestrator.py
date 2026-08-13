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
    ResearchNarrative,
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
    governance_mode = governance.get("mode")
    flagship_state = (
        "production"
        if metrics and governance_mode not in {"demoted", "retired", "degraded"}
        else "degraded"
    )
    degraded_keys = " ".join(str(key).lower() for key in (ctx.get("degraded") or {}))

    def champion(model_id: str, label: str, metric: str, aliases: tuple[str, ...]) -> ModelCard:
        degraded = any(alias in degraded_keys for alias in aliases)
        return ModelCard(
            model_id=model_id,
            label=label,
            state="degraded" if degraded else "production",
            role="champion",
            oos_metric=metric,
            degradation_reason="research_context_reported_degradation" if degraded else None,
            can_influence_strategy=not degraded,
        )

    cards = [
        champion("har_rv", "HAR-RV volatility", "coverage diagnostics", ("har", "vol_band")),
        champion("hmm", "HMM regime", "walk-forward posterior", ("hmm", "regime")),
        champion("macro_factors", "Macro factor composite", "causal factor rules", ("macro", "factor")),
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


def _strategies(
    ctx: Dict[str, Any],
    *,
    blocked: bool,
    now: datetime,
    evidence_confidence: float = 1.0,
    evidence_views: Optional[List[AgentView]] = None,
) -> Dict[str, HorizonStrategy]:
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
    strategies = {
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
    degraded_keys = " ".join(str(key).lower() for key in (ctx.get("degraded") or {}))
    model_horizon_aliases = {
        "short_term": ("har", "vol_band"),
        "mid_term": ("hmm", "regime", "macro", "factor"),
        "long_term": ("fair_value", "scenario"),
    }
    for key, aliases in model_horizon_aliases.items():
        if any(alias in degraded_keys for alias in aliases):
            strategy = strategies[key]
            strategies[key] = strategy.model_copy(update={
                "stance": "abstain" if key != "short_term" else strategy.stance,
                "confidence": min(strategy.confidence, 0.2),
                "degradation_flags": [*strategy.degradation_flags, "production_model_degraded"],
            })
    domain_views = [
        view for view in (evidence_views or [])
        if view.agent != "quant_model_risk" and view.stance in {"bullish", "bearish"}
    ]
    directional = {view.stance for view in domain_views}
    if len(directional) == 1 and evidence_confidence >= 0.7:
        evidence_stance = next(iter(directional))
        mid_strategy = strategies["mid_term"]
        if mid_strategy.stance == evidence_stance:
            strategies["mid_term"] = mid_strategy.model_copy(update={
                "confidence": min(0.8, mid_strategy.confidence * evidence_confidence + 0.08),
                "priced_in": "partial",
            })
        else:
            strategies["mid_term"] = mid_strategy.model_copy(update={
                "confidence": max(0.2, mid_strategy.confidence * evidence_confidence * 0.65),
                "priced_in": "uncertain",
                "degradation_flags": [*mid_strategy.degradation_flags, "evidence_quant_disagreement"],
            })
    elif len(directional) > 1:
        for key, strategy in strategies.items():
            strategies[key] = strategy.model_copy(update={
                "confidence": max(0.15, strategy.confidence * 0.65),
                "degradation_flags": [*strategy.degradation_flags, "agent_disagreement_preserved"],
            })
    if evidence_confidence < 1.0:
        for key, strategy in strategies.items():
            strategies[key] = strategy.model_copy(update={
                "confidence": round(strategy.confidence * max(0.25, evidence_confidence), 4),
                "degradation_flags": [*strategy.degradation_flags, "evidence_confidence_discount"],
            })
    return strategies


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
    now = now or datetime.now(timezone.utc)
    blocked = any(gate.decision == "block" for gate in evidence.gates)
    candidate_facts = evidence.accepted_facts
    all_intake_gates_pass = all(
        gate.decision == "pass" or gate.gate in {"market_coherence", "output_audit"}
        for gate in evidence.gates
    )
    accepted = candidate_facts if all_intake_gates_pass else []
    text = " ".join(fact.text for fact in accepted)
    experts = select_experts(question, text) if mode == "full" else ["quant_model_risk"]
    views = [] if blocked else []
    if not blocked:
        for expert in experts:
            view = _expert_view(expert, text, [fact.claim_id for fact in accepted], quant_ctx)
            if expert != "quant_model_risk":
                view = view.model_copy(update={
                    "confidence": round(view.confidence * evidence.confidence, 4),
                    "degradation_flags": (
                        [*view.degradation_flags, "evidence_confidence_discount"]
                        if evidence.confidence < 1.0 else view.degradation_flags
                    ),
                })
            views.append(view)
    strategies = _strategies(
        quant_ctx,
        blocked=blocked,
        now=now,
        evidence_confidence=evidence.confidence,
        evidence_views=views,
    )
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
    domain_direction = {
        view.stance for view in views
        if view.agent not in {"quant_model_risk", "strategy_arbitrator"}
        and view.stance in {"bullish", "bearish"}
    }
    market_gate_decision = "review"
    market_gate_reason = "market reaction does not yet confirm one evidence direction"
    if not accepted:
        market_gate_decision = "abstain"
        market_gate_reason = "no accepted facts are available for a market-reaction check"
    elif not stale and len(domain_direction) == 2:
        market_gate_decision = "pass"
        market_gate_reason = "opposing accepted evidence is explicitly preserved as a market-coherence conflict"
    elif not stale and len(domain_direction) == 1 and strategies["mid_term"].stance in domain_direction:
        market_gate_decision = "pass"
        market_gate_reason = "accepted evidence direction is coherent with current governed macro context"
    gates = [
        gate.model_copy(update={
            "decision": market_gate_decision,
            "reason": market_gate_reason,
            "confidence_multiplier": 1.0 if market_gate_decision == "pass" else 0.75 if market_gate_decision == "review" else 0.0,
        }) if gate.gate == "market_coherence" else gate
        for gate in evidence.gates
    ]
    case_facts = evidence.facts
    if market_gate_decision != "pass":
        case_facts = [
            fact.model_copy(update={"status": "review"})
            if fact.status == "accepted" else fact
            for fact in evidence.facts
        ]
        views = [view for view in views if view.agent == "quant_model_risk"]
        strategies = _strategies(
            quant_ctx,
            blocked=blocked,
            now=now,
            evidence_confidence=0.0,
            evidence_views=views,
        )
        if not blocked:
            views.append(AgentView(
                agent="strategy_arbitrator",
                horizon="mid_term",
                stance=strategies["mid_term"].stance,
                confidence=strategies["mid_term"].confidence,
                thesis="Unconfirmed evidence was excluded; only governed quantitative scenarios remain.",
                supporting_fact_ids=[],
                counter_fact_ids=[],
                invalidation=strategies["mid_term"].invalidation,
                degradation_flags=["market_coherence_unconfirmed"],
            ))
    nonpass = [gate for gate in gates if gate.decision != "pass"]
    status = "blocked" if blocked else "degraded" if stale or not accepted or nonpass else "complete"
    issues = [gate.reason for gate in nonpass]
    if stale:
        issues.append("quantitative context is stale")
    case_seed = f"{question}|{evidence.document.sha256}|{now.isoformat()}"
    case_id = "rc_" + hashlib.sha256(case_seed.encode("utf-8")).hexdigest()[:18]
    quant_keys = (
        "data_source", "data_asof", "data_age_days", "data_stale", "vol_bands",
        "regime_posterior", "macro_factors", "fair_value", "scenario_cone",
        "flagship", "degraded", "series_asof", "series_age_days", "stale_series",
    )
    quant_pack = {key: quant_ctx.get(key) for key in quant_keys if key in quant_ctx}
    entry_price = (
        quant_ctx.get("latest_price")
        or (quant_ctx.get("scenario_cone") or {}).get("spot")
        or (quant_ctx.get("fair_value") or {}).get("spot")
    )
    if entry_price is not None:
        try:
            entry_price = float(entry_price)
            if entry_price <= 0:
                entry_price = None
        except (TypeError, ValueError):
            entry_price = None
    if blocked:
        overview = "Evidence Shield blocked the external material, so the case preserves an abstention instead of a directional conclusion."
    elif not accepted:
        overview = "No external fact passed the evidence gates; this draft is limited to governed quantitative scenarios and must not be read as document analysis."
    elif stale:
        overview = "The evidence was structured, but stale quantitative context forces a degraded, scenario-only research conclusion."
    else:
        overview = "Accepted evidence and governed quantitative context were combined into auditable short-, medium- and long-horizon scenarios."
    narrative = ResearchNarrative(
        overview=overview,
        horizon_notes={
            horizon: strategy.base.description for horizon, strategy in strategies.items()
        },
        watchlist=[
            trigger
            for strategy in strategies.values()
            for trigger in strategy.triggers
        ][:9],
        generated_by="deterministic_draft",
    )
    return ResearchCase(
        case_id=case_id, question=question, created_at=now,
        data_asof=quant_ctx.get("data_asof"), status=status, research_mode=mode,
        evidence_documents=[evidence.document], fact_claims=case_facts,
        gate_report=gates,
        market_snapshot={
            key: quant_ctx.get(key) for key in ("latest_price", "data_asof", "data_stale")
            if key in quant_ctx
        },
        quant_pack=quant_pack, model_registry=_model_registry(quant_ctx),
        agent_views=views, conflicts=_conflicts(views), horizon_strategy=strategies,
        narrative=narrative,
        audit_report=AuditReport(passed=not blocked and not issues, issues=issues, checked_at=now),
        outcome_schedule=[
            OutcomeCheckpoint(horizon="short_term", due_at=now + timedelta(days=21), entry_price=entry_price),
            OutcomeCheckpoint(horizon="mid_term", due_at=now + timedelta(days=180), entry_price=entry_price),
            OutcomeCheckpoint(horizon="long_term", due_at=now + timedelta(days=365), entry_price=entry_price),
        ],
    )
