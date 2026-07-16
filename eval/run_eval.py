"""Run the golden-set eval and gate on quality.

Two modes:
  --live BASE_URL   : POST each case to a running gateway (needs API key)
  (default)         : run in-process against a deterministic stub toolbox, so
                      the gate works in CI with no network and no LLM key.

Exit code is non-zero if any case fails its judges, so this can guard a merge.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterator, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.judges import CaseReport, score_case, score_personal_case  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")


def load_cases(path: str = GOLDEN_PATH) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _run_case(client: Any, case: Dict[str, Any], api_key: str) -> CaseReport:
    """Route one golden case by channel: analyze (default) or personal."""
    if case.get("channel") == "personal":
        path, scorer = "/api/v1/agent/personal-research", score_personal_case
    else:
        path, scorer = "/api/v1/agent/analyze", score_case
    resp = client.post(path, headers={"X-API-Key": api_key}, json=case["request"])
    if resp.status_code != 200:
        return CaseReport(case_id=case["case_id"], passed=False, score=0.0)
    return scorer(case["case_id"], resp.json(), case.get("expect", {}))


def run_in_process(cases: List[Dict[str, Any]]) -> List[CaseReport]:
    """Drive the real pipelines with a deterministic stub toolbox/narrator."""
    os.environ.setdefault("AGENT_PUBLIC_API_KEYS", "dev-public-key")
    os.environ.setdefault("AGENT_INTERNAL_API_KEYS", "dev-internal-key")
    from fastapi.testclient import TestClient

    import agent_gateway
    from eval.stub_toolbox import StubToolbox, StubNarrator
    from signal_ledger import MemoryLedgerStore

    app = agent_gateway.create_app(
        toolbox=StubToolbox(),
        narrator=StubNarrator(),
        signal_ledger_store=MemoryLedgerStore(),
    )
    with TestClient(app) as client:
        return [_run_case(client, case, "dev-public-key") for case in cases]


def run_live(cases: List[Dict[str, Any]], base_url: str) -> List[CaseReport]:
    import httpx

    api_key = os.environ.get("AGENT_PUBLIC_API_KEYS", "dev-public-key").split(",")[0]
    with httpx.Client(base_url=base_url, timeout=40.0) as client:
        return [_run_case(client, case, api_key) for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", metavar="BASE_URL", default=None)
    args = parser.parse_args()

    cases = list(load_cases())
    reports = run_live(cases, args.live) if args.live else run_in_process(cases)

    passed = sum(1 for r in reports if r.passed)
    payload = {
        "total": len(reports),
        "passed": passed,
        "failed": len(reports) - passed,
        "cases": [r.as_dict() for r in reports],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
