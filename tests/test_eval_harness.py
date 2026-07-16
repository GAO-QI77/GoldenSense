"""Tests for the eval judges and the in-process golden-set harness."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.judges import (  # noqa: E402
    judge_disclaimer,
    judge_faithfulness,
    judge_invalidators,
    judge_risk_disclosure,
    judge_stance_expectation,
    score_case,
)


def _good_response():
    return {
        "summary_card": {
            "stance": "偏多",
            "confidence_band": "中",
            "invalidators": ["若美元指数升破 105 则看空", "若金价跌破 2300 则失效"],
            "disclaimer": "本内容为教育型研究辅助，不构成投资建议。",
            "reasons": ["多周期趋势得分 0.72，判定为 uptrend。"],
        },
        "risk_banner": {"level": "medium", "title": "注意波动", "message": "关注回撤风险与止损。"},
        "follow_up_questions": ["若 CPI 超预期该如何调整？"],
        "horizon_forecasts": [{"probability": 0.72}],
        "evidence_cards": [{"takeaway": "趋势得分 0.72"}],
        "citations": [{"excerpt": "美元指数 105"}],
        "recent_news": [{"title": "金价站上 2300"}],
    }


# ------------------------------- judges ----------------------------------- #
def test_invalidators_judge_fails_when_too_few():
    resp = _good_response()
    resp["summary_card"]["invalidators"] = ["only one"]
    assert judge_invalidators(resp, {}).passed is False
    assert judge_invalidators(_good_response(), {}).passed is True


def test_risk_disclosure_judge_requires_risk_language():
    resp = _good_response()
    resp["risk_banner"] = {"level": "low", "title": "一切正常", "message": "无需担心。"}
    assert judge_risk_disclosure(resp, {}).passed is False
    assert judge_risk_disclosure(_good_response(), {}).passed is True


def test_disclaimer_judge():
    resp = _good_response()
    resp["summary_card"]["disclaimer"] = ""
    assert judge_disclaimer(resp, {}).passed is False


def test_faithfulness_judge_flags_ungrounded_number():
    resp = _good_response()
    resp["summary_card"]["reasons"] = ["目标价 9999 美元。"]  # not in evidence
    result = judge_faithfulness(resp, {})
    assert result.passed is False
    assert result.score < 1.0


def test_stance_expectation_judge_respects_allowed_set():
    resp = _good_response()
    assert judge_stance_expectation(resp, {"allowed_stances": ["偏多", "中性"]}).passed
    assert not judge_stance_expectation(resp, {"allowed_stances": ["偏空"]}).passed
    # No expectation -> always passes.
    assert judge_stance_expectation(resp, {}).passed


def test_score_case_hard_gate_blocks_on_ungrounded_number():
    resp = _good_response()
    resp["summary_card"]["reasons"] = ["目标价 9999 美元。"]
    report = score_case("bad", resp, {})
    assert report.passed is False  # faithfulness is a hard gate


def test_score_case_passes_clean_response():
    report = score_case("good", _good_response(), {})
    assert report.passed is True
    assert report.score >= 0.8


# ------------------------------- harness ---------------------------------- #
def test_golden_set_passes_in_process():
    from eval.run_eval import load_cases, run_in_process

    cases = list(load_cases())
    assert len(cases) >= 5
    reports = run_in_process(cases)
    failed = [r.case_id for r in reports if not r.passed]
    assert not failed, f"golden cases failed: {failed}"


# --------------------------- personal channel ------------------------------ #
from eval.judges import (  # noqa: E402
    judge_no_directive_language,
    judge_personal_disclaimer,
    judge_personal_faithfulness,
    judge_risk_flags_surfaced,
    score_personal_case,
)


def _personal_response():
    return {
        "facts": {
            "reference_range": {"available": True, "range_pct": [5.0, 12.0], "midpoint": 8.5},
            "position_gap": {"status": "above", "gap_pct": 3.0, "current_gold_pct": 15.0},
            "risk_flags": [{"flag": "position_far_above_range", "detail": "超出 3.0 个百分点"}],
        },
        "narrative": {
            "overview": "基于您的稳健画像整理，仅为研究参考。",
            "position_analysis": "参考区间为 5.0%–12.0%，您的仓位 15% 高于上沿约 3.0 个百分点。",
            "risk_notes": ["高于区间时历史回撤波动更大，供您参考。"],
            "horizon_note": "中期主导状态为平静。",
            "disclaimer": "本内容为教育型研究参考，不构成投资建议。",
        },
    }


def test_personal_judges_pass_on_compliant_response():
    report = score_personal_case("ok", _personal_response(), {})
    assert report.passed is True


def test_directive_language_judge_blocks():
    resp = _personal_response()
    resp["narrative"]["position_analysis"] = "建议买入黄金到 12%。"
    assert judge_no_directive_language(resp, {}).passed is False
    assert score_personal_case("bad", resp, {}).passed is False


def test_personal_faithfulness_blocks_ungrounded_numbers():
    resp = _personal_response()
    resp["narrative"]["overview"] = "金价目标 99999 美元。"
    assert judge_personal_faithfulness(resp, {}).passed is False


def test_risk_flags_must_surface_in_notes():
    resp = _personal_response()
    resp["narrative"]["risk_notes"] = []
    assert judge_risk_flags_surfaced(resp, {}).passed is False


def test_personal_disclaimer_required():
    resp = _personal_response()
    resp["narrative"]["disclaimer"] = ""
    assert judge_personal_disclaimer(resp, {}).passed is False
