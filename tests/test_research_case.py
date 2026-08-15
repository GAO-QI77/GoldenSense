from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from research_case import (
    AgentView,
    AuditReport,
    ConfidenceBasis,
    EvidenceDocument,
    FactClaim,
    GateDecision,
    HorizonScenario,
    HorizonStrategy,
    ResearchCase,
    ResearchCaseStore,
    JsonlResearchCaseStore,
    OutcomeCheckpoint,
    score_due_outcomes,
)


def _case(case_id: str = "rc_test") -> ResearchCase:
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    document = EvidenceDocument(
        document_id="doc_1",
        kind="pdf",
        filename="fomc.pdf",
        sha256="a" * 64,
        source_url=None,
        content_type="application/pdf",
        source_tier="primary",
        retrieved_at=now,
        persisted_raw=False,
    )
    fact = FactClaim(
        claim_id="fact_1",
        document_id="doc_1",
        text="Committee maintained the target range.",
        locator="page 2",
        status="accepted",
        confidence=0.92,
        domains=["macro_event"],
    )
    return ResearchCase(
        case_id=case_id,
        question="How does this affect gold?",
        created_at=now,
        data_asof="2026-08-12",
        status="complete",
        evidence_documents=[document],
        fact_claims=[fact],
        gate_report=[
            GateDecision(
                gate="provenance_time",
                decision="pass",
                reason="primary source with timestamp",
                confidence_multiplier=1.0,
                evidence_refs=["fact_1"],
            )
        ],
        quant_pack={"data_asof": "2026-08-12"},
        agent_views=[
            AgentView(
                agent="macro_event",
                horizon="mid_term",
                stance="neutral",
                confidence=0.6,
                thesis="No new directional information.",
                supporting_fact_ids=["fact_1"],
                counter_fact_ids=[],
                invalidation=["Policy path changes."],
            )
        ],
        conflicts=[],
        horizon_strategy={
            "mid_term": HorizonStrategy(
                horizon="mid_term",
                stance="neutral",
                confidence=0.6,
                priced_in="uncertain",
                base=HorizonScenario(label="base", probability=0.6, description="Range"),
                upside=HorizonScenario(label="upside", probability=0.2, description="Real rates fall"),
                downside=HorizonScenario(label="downside", probability=0.2, description="USD rises"),
                triggers=["real yields"],
                invalidation=["new policy guidance"],
                next_review_at=now,
            )
        },
        audit_report=AuditReport(passed=True, issues=[]),
        outcome_schedule=[],
    )


def test_research_case_round_trips_json_and_never_persists_raw_documents():
    case = _case()
    restored = ResearchCase.model_validate_json(case.model_dump_json())

    assert restored.case_id == "rc_test"
    assert restored.fact_claims[0].locator == "page 2"
    assert restored.evidence_documents[0].persisted_raw is False


def test_agent_view_rejects_references_to_unaccepted_or_missing_facts():
    case = _case()
    case.fact_claims[0].status = "blocked"

    with pytest.raises(ValueError, match="accepted fact"):
        ResearchCase.model_validate(case.model_dump())


def test_domain_agent_cannot_reference_another_domain_fact():
    case = _case()
    case.fact_claims[0] = case.fact_claims[0].model_copy(update={"domains": ["technical_flows"]})

    with pytest.raises(ValueError, match="own evidence domain"):
        ResearchCase.model_validate(case.model_dump())


def test_calibrated_confidence_requires_sample_and_period():
    with pytest.raises(ValueError, match="calibrated confidence"):
        ConfidenceBasis(method="historical_reliability", calibrated=True)


def test_strategy_rejects_mixed_or_unsubstantiated_probability_kinds():
    strategy = _case().horizon_strategy["mid_term"]
    payload = strategy.model_dump()
    payload["base"]["probability_kind"] = "calibrated_probability"
    with pytest.raises(ValueError, match="same probability kind"):
        HorizonStrategy.model_validate(payload)

    for scenario in ("base", "upside", "downside"):
        payload[scenario]["probability_kind"] = "calibrated_probability"
    with pytest.raises(ValueError, match="calibrated probability"):
        HorizonStrategy.model_validate(payload)


def test_models_forbid_unknown_fields():
    with pytest.raises(ValidationError):
        GateDecision(
            gate="access",
            decision="pass",
            reason="ok",
            confidence_multiplier=1.0,
            surprise="not allowed",
        )


def test_store_is_bounded_and_returns_detached_copies():
    store = ResearchCaseStore(max_items=2)
    store.save(_case("rc_1"))
    store.save(_case("rc_2"))
    store.save(_case("rc_3"))

    assert store.get("rc_1") is None
    loaded = store.get("rc_3")
    assert loaded is not None
    loaded.question = "mutated"
    assert store.get("rc_3").question == "How does this affect gold?"


def test_gate_confidence_multiplier_is_bounded():
    with pytest.raises(ValidationError):
        GateDecision(
            gate="access",
            decision="review",
            reason="uncertain",
            confidence_multiplier=1.5,
        )


def test_jsonl_store_is_append_only_and_loads_latest_revision(tmp_path):
    path = tmp_path / "research_cases.jsonl"
    store = JsonlResearchCaseStore(path, max_items=5)
    original = _case("rc_append")
    store.save(original)
    store.save(original.model_copy(update={"personalized_brief": {"case_id": "rc_append"}}))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all('"personalized_brief":null' in line for line in lines)
    assert store.get("rc_append").personalized_brief is None

    restored = JsonlResearchCaseStore(path, max_items=5)
    assert restored.get("rc_append").personalized_brief is None


def test_due_outcomes_are_scored_forward_without_touching_future_checkpoints():
    case = _case("rc_score")
    published = case.created_at
    case.horizon_strategy["short_term"] = case.horizon_strategy["mid_term"].model_copy(
        update={
            "horizon": "short_term",
            "stance": "risk",
            "next_review_at": published,
        }
    )
    case.outcome_schedule = [
        OutcomeCheckpoint(
            horizon="short_term", due_at=published, status="scheduled", entry_price=3000.0,
        ),
        OutcomeCheckpoint(
            horizon="long_term", due_at=datetime(2027, 8, 13, tzinfo=timezone.utc),
            status="scheduled", entry_price=3000.0,
        ),
    ]

    scored = score_due_outcomes(case, as_of=published, price_at=lambda _: 3060.0)

    checkpoint = scored.outcome_schedule[0]
    assert checkpoint.scenario_outcome == "upside"
    assert checkpoint.scenario_score is not None
    assert checkpoint.scenario_score_kind == "weight_brier"
    assert checkpoint.neutral_band_pct == pytest.approx(0.015)
    assert checkpoint.confidence_error is None  # risk/abstain has no direction claim
    short, future = scored.outcome_schedule
    assert short.status == "scored"
    assert short.realized_return == pytest.approx(0.02)
    assert short.direction_score is None
    assert future.status == "scheduled" and future.realized_return is None


def test_outcome_scoring_uses_horizon_specific_neutral_bands():
    case = _case("rc_horizon_bands")
    published = case.created_at
    case.horizon_strategy["short_term"] = case.horizon_strategy["mid_term"].model_copy(
        update={"horizon": "short_term", "stance": "neutral"}
    )
    case.horizon_strategy["long_term"] = case.horizon_strategy["mid_term"].model_copy(
        update={"horizon": "long_term", "stance": "neutral"}
    )
    case.outcome_schedule = [
        OutcomeCheckpoint(horizon="short_term", due_at=published, entry_price=100.0),
        OutcomeCheckpoint(horizon="long_term", due_at=published, entry_price=100.0),
    ]

    scored = score_due_outcomes(case, as_of=published, price_at=lambda _: 106.0)

    assert scored.outcome_schedule[0].direction_score == 0.0
    assert scored.outcome_schedule[1].direction_score == 1.0
    assert scored.outcome_schedule[0].neutral_band_pct < scored.outcome_schedule[1].neutral_band_pct


def test_abstention_weights_are_not_scored_as_forecast_probabilities():
    case = _case("rc_abstain_score")
    published = case.created_at
    strategy = case.horizon_strategy["mid_term"]
    case.horizon_strategy["mid_term"] = strategy.model_copy(update={
        "stance": "abstain",
        "base": strategy.base.model_copy(update={"probability": 1.0, "probability_kind": "abstention"}),
        "upside": strategy.upside.model_copy(update={"probability": 0.0, "probability_kind": "abstention"}),
        "downside": strategy.downside.model_copy(update={"probability": 0.0, "probability_kind": "abstention"}),
    })
    case.outcome_schedule = [
        OutcomeCheckpoint(horizon="mid_term", due_at=published, entry_price=100.0),
    ]

    scored = score_due_outcomes(case, as_of=published, price_at=lambda _: 110.0)

    assert scored.outcome_schedule[0].scenario_score is None
    assert scored.outcome_schedule[0].scenario_score_kind is None

def test_forward_scoring_uses_each_checkpoint_due_date_price():
    case = _case("rc_due_prices")
    start = case.created_at
    case.outcome_schedule = [
        OutcomeCheckpoint(horizon="short_term", due_at=start, entry_price=100.0),
        OutcomeCheckpoint(horizon="mid_term", due_at=start + timedelta(days=1), entry_price=100.0),
    ]
    prices = {start: 101.0, start + timedelta(days=1): 110.0}

    scored = score_due_outcomes(
        case,
        as_of=start + timedelta(days=2),
        price_at=lambda due_at: prices.get(due_at),
    )

    assert [item.realized_return for item in scored.outcome_schedule] == pytest.approx([0.01, 0.10])


def test_matured_checkpoint_retries_when_due_date_price_arrives_later():
    case = _case("rc_retry_price")
    due = case.created_at
    case.outcome_schedule = [OutcomeCheckpoint(horizon="short_term", due_at=due, entry_price=100.0)]
    waiting = score_due_outcomes(case, as_of=due, price_at=lambda _: None)
    assert waiting.outcome_schedule[0].status == "matured"

    scored = score_due_outcomes(waiting, as_of=due + timedelta(days=1), price_at=lambda _: 102.0)
    assert scored.outcome_schedule[0].status == "scored"
    assert scored.outcome_schedule[0].realized_return == pytest.approx(0.02)
