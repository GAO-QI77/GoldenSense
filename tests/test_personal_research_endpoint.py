"""Contract tests for POST /api/v1/agent/personal-research.

The narrator is injected per-app so tests exercise all three gate outcomes:
LLM unavailable (draft), ungrounded numbers (critic revert), directive
language (compliance revert).
"""
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AGENT_PUBLIC_API_KEYS", "dev-public-key")
os.environ.setdefault("AGENT_INTERNAL_API_KEYS", "dev-internal-key")

import agent_gateway  # noqa: E402
from investor_profile import PersonalNarrative  # noqa: E402
from signal_ledger import MemoryLedgerStore  # noqa: E402

PUBLIC_HEADERS = {"X-API-Key": "dev-public-key"}

VALID_BODY = {
    "risk_tolerance": "balanced",
    "horizon": "mid",
    "current_gold_pct": 15.0,
    "experience": "novice",
}


class _DraftEchoNarrator(agent_gateway.OpenAINarrator):
    """Simulates the no-API-key environment: always degrades to the draft."""

    def __init__(self):  # bypass client construction
        pass

    async def narrate(self, bundle, draft):
        return draft

    async def narrate_personal(self, facts, profile, draft):
        return draft


class _UngroundedNarrator(_DraftEchoNarrator):
    async def narrate_personal(self, facts, profile, draft):
        return PersonalNarrative(
            overview="金价即将上看 99999 美元。",
            position_analysis=draft.position_analysis,
            risk_notes=list(draft.risk_notes),
            horizon_note=draft.horizon_note,
            disclaimer=draft.disclaimer,
        )


class _DirectiveNarrator(_DraftEchoNarrator):
    async def narrate_personal(self, facts, profile, draft):
        return PersonalNarrative(
            overview=draft.overview,
            position_analysis="建议买入黄金，把仓位加到区间上限。",
            risk_notes=list(draft.risk_notes),
            horizon_note=draft.horizon_note,
            disclaimer=draft.disclaimer,
        )


def _client(narrator) -> TestClient:
    app = agent_gateway.create_app(
        narrator=narrator, signal_ledger_store=MemoryLedgerStore()
    )
    return TestClient(app)


def test_requires_api_key():
    with _client(_DraftEchoNarrator()) as client:
        assert client.post(
            "/api/v1/agent/personal-research", json=VALID_BODY
        ).status_code in (401, 403)


def test_invalid_profile_is_422():
    with _client(_DraftEchoNarrator()) as client:
        resp = client.post(
            "/api/v1/agent/personal-research",
            headers=PUBLIC_HEADERS,
            json={**VALID_BODY, "risk_tolerance": "yolo"},
        )
        assert resp.status_code == 422


def test_draft_path_contract():
    with _client(_DraftEchoNarrator()) as client:
        resp = client.post(
            "/api/v1/agent/personal-research", headers=PUBLIC_HEADERS, json=VALID_BODY
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["generated_by"] == "deterministic_draft"
        assert payload["narrative"]["disclaimer"]
        facts = payload["facts"]
        assert facts["reference_range"]["available"] in (True, False)
        assert facts["position_gap"]["current_gold_pct"] == 15.0
        assert "risk_flags" in facts and "horizon_evidence" in facts
        # No personal data may be persisted: response echoes but stores nothing.
        assert payload["profile_echo"] == VALID_BODY


def test_ungrounded_llm_output_reverts_to_draft():
    with _client(_UngroundedNarrator()) as client:
        payload = client.post(
            "/api/v1/agent/personal-research", headers=PUBLIC_HEADERS, json=VALID_BODY
        ).json()
        assert "narrative_critic_reverted" in payload["degradation_flags"]
        assert payload["generated_by"] == "deterministic_draft"
        assert "99999" not in payload["narrative"]["overview"]


def test_directive_language_reverts_to_draft():
    with _client(_DirectiveNarrator()) as client:
        payload = client.post(
            "/api/v1/agent/personal-research", headers=PUBLIC_HEADERS, json=VALID_BODY
        ).json()
        assert "directive_language_reverted" in payload["degradation_flags"]
        assert "建议买入" not in payload["narrative"]["position_analysis"]


class _ExplodingNarrator(_DraftEchoNarrator):
    """Fails the test if the endpoint calls the LLM in draft mode."""

    async def narrate_personal(self, facts, profile, draft):
        raise AssertionError("narrate_personal must not be called in draft mode")


def test_draft_mode_skips_llm_and_returns_fast():
    with _client(_ExplodingNarrator()) as client:
        resp = client.post(
            "/api/v1/agent/personal-research?mode=draft",
            headers=PUBLIC_HEADERS,
            json=VALID_BODY,
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["mode"] == "draft"
        assert payload["generated_by"] == "deterministic_draft"
        assert payload["narrative"]["disclaimer"]
        assert payload["facts"]["reference_range"]


def test_full_mode_is_default_and_unchanged():
    with _client(_DraftEchoNarrator()) as client:
        payload = client.post(
            "/api/v1/agent/personal-research",
            headers=PUBLIC_HEADERS,
            json=VALID_BODY,
        ).json()
        assert payload["mode"] == "full"


def test_advanced_profile_accepted_by_endpoint():
    body = {**VALID_BODY, "max_drawdown_pct": 1.0, "leverage_attitude": "high"}
    with _client(_DraftEchoNarrator()) as client:
        payload = client.post(
            "/api/v1/agent/personal-research?mode=draft",
            headers=PUBLIC_HEADERS,
            json=body,
        ).json()
        flags = {f["flag"] for f in payload["facts"]["risk_flags"]}
        assert "leverage_out_of_scope" in flags
