import asyncio
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from reportlab.pdfgen import canvas

from evidence_shield import ingest_evidence
from research_orchestrator import build_research_case


def _pdf(text: str) -> bytes:
    output = BytesIO()
    doc = canvas.Canvas(output)
    doc.drawString(30, 760, text[:150])
    doc.save()
    return output.getvalue()


def _multipage_pdf(lines_by_page: list[list[str]]) -> bytes:
    output = BytesIO()
    doc = canvas.Canvas(output)
    for lines in lines_by_page:
        y = 760
        for line in lines:
            doc.drawString(30, y, line)
            y -= 26
        doc.showPage()
    doc.save()
    return output.getvalue()


def _ctx():
    return {
        "data_asof": "2026-08-12", "data_age_days": 1, "data_stale": False,
        "vol_bands": {
            "h1": {"p10": -0.01, "p50": 0, "p90": 0.01},
            "h5": {"p10": -0.02, "p50": 0, "p90": 0.02},
            "h21": {"p10": -0.05, "p50": 0.01, "p90": 0.06, "ann_vol_forecast": 0.18},
        },
        "regime_posterior": {"latest": {"calm": 0.2, "elevated": 0.7, "stress": 0.1}},
        "macro_factors": {"composite": 0.61, "factors_used": ["real_rate_momentum"]},
        "fair_value": {"deviation_pct": 5, "deviation_z": 1, "regime_break": False},
        "scenario_cone": {"checkpoints": {}}, "degraded": {},
    }


def test_clean_attacked_pairs_preserve_clean_path_and_gate_attacks():
    cases = json.loads(
        (Path(__file__).parents[1] / "eval" / "research_case_redteam.json").read_text(encoding="utf-8")
    )
    now = datetime(2026, 8, 13, tzinfo=timezone.utc)
    for fixture in cases:
        clean = asyncio.run(ingest_evidence(
            question="gold impact", filename="clean.pdf", content_type="application/pdf",
            payload=_pdf(fixture["clean"]), now=now,
        ))
        attacked = asyncio.run(ingest_evidence(
            question="gold impact", filename="attacked.pdf", content_type="application/pdf",
            payload=_pdf(fixture["attacked"]), now=now,
        ))
        assert clean.accepted_facts, fixture["id"]
        decisions = {gate.gate: gate.decision for gate in attacked.gates}
        assert fixture["expected"] in decisions.values(), fixture["id"]

        clean_case = build_research_case("gold impact", clean, _ctx(), now=now)
        attacked_case = build_research_case("gold impact", attacked, _ctx(), now=now)
        if fixture["expected"] == "block":
            assert attacked_case.status == "blocked"
            assert all(strategy.stance == "abstain" for strategy in attacked_case.horizon_strategy.values())
        else:
            assert attacked.confidence < clean.confidence
            assert attacked_case.horizon_strategy["short_term"].stance in {"risk", "abstain"}
        assert clean_case.status in {"complete", "degraded"}


def test_repeated_heading_downgrades_only_its_claim_cluster():
    packet = asyncio.run(
        ingest_evidence(
            question="gold research",
            filename="benign.pdf",
            content_type="application/pdf",
            payload=_multipage_pdf(
                [
                    [
                        "Market Research Summary.",
                        "Federal Reserve rate cuts lowered real yields and supported gold demand.",
                    ],
                    [
                        "Market Research Summary.",
                        "Central bank reserve diversification increased structural gold demand.",
                    ],
                ]
            ),
            now=datetime(2026, 8, 13, tzinfo=timezone.utc),
        )
    )

    accepted = [fact for fact in packet.facts if fact.status == "accepted"]
    reviewed = [fact for fact in packet.facts if fact.status == "review"]
    assert len(accepted) == 2
    assert len(reviewed) == 1
    assert reviewed[0].text == "Market Research Summary."
    assert next(g for g in packet.gates if g.gate == "dedup_replay").decision == "review"
    assert next(g for g in packet.gates if g.gate == "output_audit").decision == "pass"


def test_output_audit_abstains_when_every_claim_is_a_duplicate_cluster():
    packet = asyncio.run(
        ingest_evidence(
            question="gold research",
            filename="duplicates.pdf",
            content_type="application/pdf",
            payload=_multipage_pdf(
                [["Repeated market statement has no unique supporting detail."],
                 ["Repeated market statement has no unique supporting detail."]]
            ),
            now=datetime(2026, 8, 13, tzinfo=timezone.utc),
        )
    )

    assert packet.accepted_facts == []
    output_gate = next(g for g in packet.gates if g.gate == "output_audit")
    assert output_gate.decision == "abstain"
    assert "accepted claims are traceable" not in output_gate.reason


def test_competition_rules_route_to_non_directional_product_review_agent():
    packet = asyncio.run(
        ingest_evidence(
            question="Review the competition rules",
            filename="rules.pdf",
            content_type="application/pdf",
            payload=_pdf(
                "Competition rules require a three-minute demo, evidence citations, and security compliance."
            ),
            now=datetime(2026, 8, 13, tzinfo=timezone.utc),
        )
    )
    case = build_research_case("Review the competition rules", packet, _ctx())

    rule_view = next(view for view in case.agent_views if view.agent == "product_rules_review")
    assert rule_view.stance == "abstain"
    assert rule_view.supporting_fact_ids
    arbitrator = next(view for view in case.agent_views if view.agent == "strategy_arbitrator")
    assert not set(rule_view.supporting_fact_ids) & set(arbitrator.supporting_fact_ids)
