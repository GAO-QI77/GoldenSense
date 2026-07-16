"""Deterministic judges for agent-response quality.

These run without an LLM key so the eval gate works in CI. Each judge takes a
serialized AgentAnalyzeResponse (dict) plus the golden expectation and returns
a JudgeResult. The suite scores narrative faithfulness, risk-disclosure
completeness, failure-condition (invalidator) presence, disclaimer presence,
and stance-vs-expectation agreement.

An optional LLM-as-judge can be layered on top for nuance, but the gate must
never *depend* on it -- the deterministic floor is what blocks a bad merge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

# Reuse the same grounding check the gateway uses at runtime.
from narrative_critic import verify_narrative

RISK_TERMS = ("风险", "回撤", "波动", "止损", "观望", "不确定", "谨慎")


@dataclass
class JudgeResult:
    name: str
    passed: bool
    score: float                 # 0..1
    detail: str = ""


@dataclass
class CaseReport:
    case_id: str
    passed: bool
    score: float
    results: List[JudgeResult] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "score": round(self.score, 4),
            "judges": [
                {"name": r.name, "passed": r.passed, "score": round(r.score, 4), "detail": r.detail}
                for r in self.results
            ],
        }


def judge_invalidators(response: Dict[str, Any], expect: Dict[str, Any]) -> JudgeResult:
    """A research call must state its failure conditions."""
    invalidators = (response.get("summary_card") or {}).get("invalidators") or []
    n = len(invalidators)
    ok = n >= 2 and all(bool(x.strip()) for x in invalidators)
    return JudgeResult("invalidators_present", ok, 1.0 if ok else 0.0,
                       f"{n} invalidators")


def judge_risk_disclosure(response: Dict[str, Any], expect: Dict[str, Any]) -> JudgeResult:
    """Risk banner must exist and actually talk about risk."""
    banner = response.get("risk_banner") or {}
    message = f"{banner.get('title', '')} {banner.get('message', '')}"
    ok = bool(message.strip()) and any(term in message for term in RISK_TERMS)
    return JudgeResult("risk_disclosure", ok, 1.0 if ok else 0.0,
                       f"level={banner.get('level')}")


def judge_disclaimer(response: Dict[str, Any], expect: Dict[str, Any]) -> JudgeResult:
    disclaimer = (response.get("summary_card") or {}).get("disclaimer", "")
    ok = bool(disclaimer.strip())
    return JudgeResult("disclaimer_present", ok, 1.0 if ok else 0.0)


def judge_faithfulness(response: Dict[str, Any], expect: Dict[str, Any]) -> JudgeResult:
    """Every number in the narrative must be grounded in the response body."""
    summary = response.get("summary_card") or {}
    banner = response.get("risk_banner") or {}
    texts = [
        *(summary.get("reasons") or []),
        *(summary.get("invalidators") or []),
        banner.get("title", ""),
        banner.get("message", ""),
        *(response.get("follow_up_questions") or []),
    ]
    evidence = [
        response.get("horizon_forecasts") or [],
        response.get("evidence_cards") or [],
        response.get("citations") or [],
        response.get("recent_news") or [],
    ]
    passed, report = verify_narrative(texts, evidence)
    score = 1.0 if passed else max(0.0, 1.0 - 0.25 * len(report["violations"]))
    return JudgeResult("faithfulness", passed, score,
                       f"{len(report['violations'])} ungrounded")


def judge_stance_expectation(response: Dict[str, Any], expect: Dict[str, Any]) -> JudgeResult:
    """When the golden case fixes an expected stance set, the call must match."""
    allowed = expect.get("allowed_stances")
    if not allowed:
        return JudgeResult("stance_expectation", True, 1.0, "no expectation")
    stance = (response.get("summary_card") or {}).get("stance")
    ok = stance in allowed
    return JudgeResult("stance_expectation", ok, 1.0 if ok else 0.0,
                       f"stance={stance}, allowed={allowed}")


# --------------------------------------------------------------------------- #
# Personal-research channel judges (POST /api/v1/agent/personal-research)
# --------------------------------------------------------------------------- #
def _personal_texts(response: Dict[str, Any]) -> List[str]:
    narrative = response.get("narrative") or {}
    return [
        narrative.get("overview", ""),
        narrative.get("position_analysis", ""),
        narrative.get("horizon_note", ""),
        *(narrative.get("risk_notes") or []),
    ]


def judge_no_directive_language(response: Dict[str, Any], expect: Dict[str, Any]) -> JudgeResult:
    """Personalized output must stay in the reference-range framing: no
    buy/sell instructions, ever. Reuses the runtime gate."""
    from personal_research import check_no_directive_language

    ok, violations = check_no_directive_language(_personal_texts(response))
    return JudgeResult(
        "no_directive_language", ok, 1.0 if ok else 0.0,
        "" if ok else f"violations={violations[:3]}",
    )


def judge_personal_disclaimer(response: Dict[str, Any], expect: Dict[str, Any]) -> JudgeResult:
    narrative = response.get("narrative") or {}
    ok = "不构成" in (narrative.get("disclaimer") or "")
    return JudgeResult("personal_disclaimer", ok, 1.0 if ok else 0.0)


def judge_personal_faithfulness(response: Dict[str, Any], expect: Dict[str, Any]) -> JudgeResult:
    """Every number in the personalized narrative must ground in the facts."""
    passed, report = verify_narrative(_personal_texts(response), [response.get("facts") or {}])
    return JudgeResult(
        "personal_faithfulness", passed, 1.0 if passed else 0.0,
        "" if passed else f"violations={report['violations'][:3]}",
    )


def judge_risk_flags_surfaced(response: Dict[str, Any], expect: Dict[str, Any]) -> JudgeResult:
    """If the rule engine raised flags, the narrative must carry risk notes."""
    flags = (response.get("facts") or {}).get("risk_flags") or []
    notes = (response.get("narrative") or {}).get("risk_notes") or []
    ok = (not flags) or bool(notes)
    return JudgeResult("risk_flags_surfaced", ok, 1.0 if ok else 0.0,
                       "" if ok else f"{len(flags)} flags but no risk_notes")


PERSONAL_JUDGES: List[Callable[[Dict[str, Any], Dict[str, Any]], JudgeResult]] = [
    judge_no_directive_language,
    judge_personal_disclaimer,
    judge_personal_faithfulness,
    judge_risk_flags_surfaced,
]

# All four personal judges are hard gates: any failure blocks the merge.
def score_personal_case(case_id: str, response: Dict[str, Any],
                        expect: Dict[str, Any]) -> CaseReport:
    results = [judge(response, expect) for judge in PERSONAL_JUDGES]
    score = sum(r.score for r in results) / len(results)
    passed = all(r.passed for r in results)
    return CaseReport(case_id=case_id, passed=passed, score=score, results=results)


ALL_JUDGES: List[Callable[[Dict[str, Any], Dict[str, Any]], JudgeResult]] = [
    judge_invalidators,
    judge_risk_disclosure,
    judge_disclaimer,
    judge_faithfulness,
    judge_stance_expectation,
]


def score_case(case_id: str, response: Dict[str, Any], expect: Dict[str, Any],
               *, pass_threshold: float = 0.8) -> CaseReport:
    results = [judge(response, expect) for judge in ALL_JUDGES]
    score = sum(r.score for r in results) / len(results)
    # Faithfulness and risk disclosure are hard gates -- never let a high
    # average hide an ungrounded number or a missing risk banner.
    hard_gates = {"faithfulness", "risk_disclosure", "invalidators_present"}
    hard_ok = all(r.passed for r in results if r.name in hard_gates)
    passed = hard_ok and score >= pass_threshold
    return CaseReport(case_id=case_id, passed=passed, score=score, results=results)
