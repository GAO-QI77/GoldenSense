"""Deterministic narrative critic: every number the LLM says must be grounded.

The narrator LLM only reorganizes evidence, but "only narrates" is a promise
that needs a gate, not a comment. This critic runs after narration and before
the response leaves the gateway:

1. Extract every numeric claim (prices, percentages, scores) from the
   narrative text fields.
2. Collect every number present in the evidence bundle (snapshot, forecast,
   regime, committee, macro context, news sentiment...).
3. A narrative number is *grounded* if it matches some evidence number within
   a small relative tolerance. Ungrounded numbers -> the critic fails and the
   gateway falls back to the deterministic draft narrative, recording
   ``narrative_critic_reverted`` in the degradation flags.

No LLM is used to check the LLM; the check itself must be auditable.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Set, Tuple

# Numbers like 4025, 4,025.5, 3.2%, -0.7, ４０２５ are matched after
# normalization; percent signs are captured so 3.2% can also ground "3.2".
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Small integers appear naturally in prose ("3 个证据", "24 小时", "T+7") and
# in enumerations; below this magnitude we do not require grounding.
_MIN_MAGNITUDE = 10.0
_REL_TOL = 0.02
_ABS_TOL = 0.51  # rounded display values ("约 4024") still ground


def extract_numbers(text: str) -> List[float]:
    values: List[float] = []
    for match in _NUM_RE.findall(text or ""):
        cleaned = match.replace(",", "")
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values


def _walk_numbers(node: Any, out: Set[float]) -> None:
    if node is None or isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        value = float(node)
        if math.isfinite(value):
            out.add(value)
            out.add(round(value, 1))
            out.add(round(value, 2))
        return
    if isinstance(node, str):
        for value in extract_numbers(node):
            out.add(value)
        return
    if isinstance(node, dict):
        for child in node.values():
            _walk_numbers(child, out)
        return
    if isinstance(node, (list, tuple)):
        for child in node:
            _walk_numbers(child, out)


def collect_evidence_numbers(evidence_payloads: Iterable[Any]) -> Set[float]:
    out: Set[float] = set()
    for payload in evidence_payloads:
        _walk_numbers(payload, out)
    return out


def _is_grounded(value: float, evidence: Set[float]) -> bool:
    if abs(value) < _MIN_MAGNITUDE:
        return True
    for ref in evidence:
        if abs(value - ref) <= max(_ABS_TOL, abs(ref) * _REL_TOL):
            return True
        # A narrative "3.2%" may ground against evidence stored as 0.032.
        if ref != 0 and abs(value - ref * 100.0) <= max(_ABS_TOL, abs(ref * 100.0) * _REL_TOL):
            return True
    return False


def verify_narrative(
    narrative_texts: Iterable[str],
    evidence_payloads: Iterable[Any],
) -> Tuple[bool, Dict[str, Any]]:
    """Return (passed, report). ``report`` lists every ungrounded claim."""
    evidence = collect_evidence_numbers(evidence_payloads)
    violations: List[Dict[str, Any]] = []
    checked = 0
    for text in narrative_texts:
        for value in extract_numbers(text or ""):
            if abs(value) < _MIN_MAGNITUDE:
                continue
            checked += 1
            if not _is_grounded(value, evidence):
                violations.append({"value": value, "text": (text or "")[:120]})

    passed = not violations
    return passed, {
        "passed": passed,
        "numbers_checked": checked,
        "violations": violations[:10],
        "evidence_number_count": len(evidence),
    }
