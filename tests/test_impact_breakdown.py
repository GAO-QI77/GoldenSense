from __future__ import annotations

from agent_gateway import AgentDecision, RagEventItem, compute_impact_breakdown


def test_impact_breakdown_shape_and_ranges():
    decision = AgentDecision(
        action="SELL",
        confidence=0.8,
        horizon="T+1",
        reasoning_summary="x",
        risk_warning="y",
    )
    rag_events = [
        RagEventItem(headline="a", similarity=0.7, gold_t1_return=-0.01, gold_t7_return=-0.02),
        RagEventItem(headline="b", similarity=0.6, gold_t1_return=-0.02, gold_t7_return=-0.01),
        RagEventItem(headline="c", similarity=0.5, gold_t1_return=0.01, gold_t7_return=0.0),
    ]
    risk_result = {"decision": "REJECTED", "executed_position": 0.0, "current_vix": 35.2, "vix_threshold": 30.0, "notes": "Circuit Breaker"}

    impact = compute_impact_breakdown(
        emotion_weight=0.9,
        rag_events=rag_events,
        decision=decision,
        xgboost_probability=0.2,
        risk_result=risk_result,
    )

    d = impact.model_dump()
    assert set(d.keys()) == {"emotion_weight", "rag_consistency", "quant_consistency", "risk_reason"}
    assert -1.0 <= d["emotion_weight"] <= 1.0
    assert 0.0 <= d["rag_consistency"] <= 1.0
    assert 0.0 <= d["quant_consistency"] <= 1.0
    assert isinstance(d["risk_reason"], str)
    assert len(d["risk_reason"]) <= 200

