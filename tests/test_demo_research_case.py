from scripts import demo_research_case


def test_demo_profile_is_suitability_complete_and_unlevered():
    assert hasattr(demo_research_case, "_demo_profile")
    profile = demo_research_case._demo_profile()

    assert profile["portfolio_context_known"] is True
    assert profile["gold_instrument"] == "unlevered_etf"
    assert profile["leverage_attitude"] == "none"
    assert profile["jurisdiction"]
    assert profile["base_currency"]


def test_clean_demo_material_covers_three_expert_domains_without_attack_language():
    assert hasattr(demo_research_case, "_clean_demo_text")
    text = demo_research_case._clean_demo_text().lower()

    assert "real yields fell" in text
    assert "etf outflows" in text
    assert "central bank demand" in text
    assert "ignore previous" not in text


def test_demo_summary_exposes_numeric_provenance_and_agent_evidence():
    payload = {
        "case_id": "rc_demo",
        "status": "complete",
        "gate_report": [{"gate": "access", "decision": "pass"}],
        "fact_claims": [{"claim_id": "f1", "domains": ["macro_event"], "status": "accepted"}],
        "agent_views": [{
            "agent": "macro_event", "stance": "bullish", "confidence": 0.6,
            "supporting_fact_ids": ["f1"], "counter_fact_ids": [],
            "confidence_basis": {"method": "deterministic_evidence_score", "calibrated": False},
        }],
        "horizon_strategy": {"mid_term": {
            "stance": "bullish", "confidence": 0.58,
            "confidence_basis": {"method": "model_governance_score", "calibrated": False},
            "base": {"probability": 0.5, "probability_kind": "research_weight", "method": "macro_composite_weight_v1"},
            "upside": {"probability": 0.31}, "downside": {"probability": 0.19},
        }},
        "outcome_schedule": [{"horizon": "mid_term", "status": "scheduled"}],
    }

    assert hasattr(demo_research_case, "_summarize_case")
    summary = demo_research_case._summarize_case(payload)

    assert summary["facts_by_domain"]["macro_event"] == 1
    assert summary["agent_evidence"][0]["supporting_facts"] == 1
    assert summary["strategy"]["mid_term"]["number_kind"] == "research_weight"
    assert summary["strategy"]["mid_term"]["method"] == "macro_composite_weight_v1"
