from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from research_case import (
    AgentView,
    AuditReport,
    EvidenceDocument,
    FactClaim,
    GateDecision,
    HorizonScenario,
    HorizonStrategy,
    ResearchCase,
    ResearchCaseStore,
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
