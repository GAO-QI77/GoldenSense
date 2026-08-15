from scripts.competition_acceptance import evaluate_competition_evidence


def _passing_evidence() -> dict:
    return {
        "backend_tests": {"exit_code": 0, "passed": 420},
        "frontend_unit": {"exit_code": 0, "passed": 7},
        "frontend_build": {"exit_code": 0},
        "playwright": {"exit_code": 0, "passed": 60},
        "market_concurrency": {
            "requests": 100,
            "successes": 100,
            "unique_instrument_price_pairs": 1,
            "identity_violations": 0,
        },
        "public_horizons": {
            "observed": ["short_term", "mid_term", "long_term"],
            "legacy_public_matches": 0,
        },
        "security": {
            "clean_document_accepted": 2,
            "prompt_injection_visible": True,
            "unit_attack_contained": True,
            "instrument_attack_quarantined": True,
        },
        "public_language": {"samples": 20, "directive_violations": 0},
        "connected_pages": {"expected": 4, "successful": 4, "rate_limit_errors": 0},
        "ledger": {
            "evidence_classes": ["backtest", "simulated_forward", "live_forward"],
            "append_only_verified": True,
            "empty_state_guidance": True,
        },
        "evidence_chain": {
            "eight_gates": True,
            "primary_source_integrity": True,
            "product_rules_isolated": True,
        },
        "agent_model_governance": {
            "challengers_locked": True,
            "disagreements_preserved": True,
            "three_horizon_strategy": True,
            "three_dimensional_personalization": True,
        },
        "mobile": {"min_touch_target_px": 44, "horizontal_overflow": False},
        "operations": {"health_ready": True, "retry_after_verified": True},
        "industry_scene": {
            "target_users_defined": True,
            "problem_defined": True,
            "value_validated": True,
            "replication_potential": True,
            "gold_specialization": True,
        },
        "openness": {
            "readme_complete": True,
            "deployment_documented": True,
            "tests_reproducible": True,
            "license_and_dependencies_disclosed": True,
        },
    }


def test_passing_evidence_scores_above_ninety_with_six_categories():
    report = evaluate_competition_evidence(_passing_evidence())

    assert report["score"] > 90
    assert all(report["hard_gates"].values())
    assert set(report["category_scores"]) == {
        "industry_scene_value",
        "agent_task_loop",
        "product_demo",
        "technical_depth",
        "safety_traceability",
        "openness_reuse",
    }
    assert report["evidence"]
    assert "remaining_risks" in report


def test_any_failed_hard_gate_caps_score_at_eighty_nine():
    evidence = _passing_evidence()
    evidence["market_concurrency"]["unique_instrument_price_pairs"] = 2

    report = evaluate_competition_evidence(evidence)

    assert report["hard_gates"]["deterministic_market"] is False
    assert report["score"] <= 89
    assert report["uncapped_score"] > report["score"]


def test_unvalidated_user_value_loses_official_five_points_without_failing_hard_gates():
    evidence = _passing_evidence()
    evidence["industry_scene"]["value_validated"] = False

    report = evaluate_competition_evidence(evidence)

    assert report["score"] == 95
    assert report["category_scores"]["industry_scene_value"] == 20
    assert all(report["hard_gates"].values())


def test_missing_measurements_never_receive_implicit_credit():
    report = evaluate_competition_evidence({})

    assert report["score"] == 0
    assert not any(report["hard_gates"].values())
    assert report["remaining_risks"]
