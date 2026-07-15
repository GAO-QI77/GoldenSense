"""Tests for the research/calibration endpoints and the committee wiring."""
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AGENT_PUBLIC_API_KEYS", "dev-public-key")
os.environ.setdefault("AGENT_INTERNAL_API_KEYS", "dev-internal-key")

import agent_gateway  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app = agent_gateway.create_app()
    with TestClient(app) as test_client:
        yield test_client


PUBLIC_HEADERS = {"X-API-Key": "dev-public-key"}


def test_research_current_requires_api_key(client):
    resp = client.get("/api/v1/agent/research/current")
    assert resp.status_code in (401, 403)


def test_research_current_returns_quant_blocks(client):
    resp = client.get("/api/v1/agent/research/current", headers=PUBLIC_HEADERS)
    assert resp.status_code == 200
    payload = resp.json()
    # Every block is either present or explicitly listed as degraded.
    expected_blocks = {
        "regime_posterior",
        "fair_value",
        "vol_bands",
        "macro_factors",
        "scenario_cone",
        "allocation",
    }
    for block in expected_blocks:
        assert block in payload or block in payload.get("degraded", {}), block
    assert "data_source" in payload or "market_data" in payload.get("degraded", {})


def test_research_current_reports_data_freshness(client):
    payload = client.get(
        "/api/v1/agent/research/current", headers=PUBLIC_HEADERS
    ).json()
    if "market_data" in payload.get("degraded", {}):
        pytest.skip("market data degraded in this environment")
    # The quant page must never imply real-time data; freshness must be explicit.
    assert payload["is_realtime"] is False
    assert "data_asof" in payload
    assert isinstance(payload["data_age_days"], int)
    assert isinstance(payload["data_stale"], bool)
    assert payload["data_source"] in {"extended", "base"}


def test_research_current_regime_posterior_shape(client):
    payload = client.get(
        "/api/v1/agent/research/current", headers=PUBLIC_HEADERS
    ).json()
    regime = payload.get("regime_posterior")
    if regime is None:
        pytest.skip("regime posterior degraded in this environment")
    latest = regime["latest"]
    assert set(latest) == {"calm", "elevated", "stress"}
    assert abs(sum(latest.values()) - 1.0) < 0.02


def test_calibration_endpoint_contract(client):
    resp = client.get("/api/v1/agent/calibration", headers=PUBLIC_HEADERS)
    assert resp.status_code == 200
    payload = resp.json()
    for key in ("total_scored", "directional_calls", "neutral_or_gated",
                "weight_adjustment", "recent_outcomes"):
        assert key in payload
    assert payload["weight_adjustment"]["fused_confidence_multiplier"] >= 0.8


def test_analyze_response_trace_contains_committee():
    # The analyze path needs downstream tools; reuse the scenario toolbox
    # from the main analyze test-suite instead of live services.
    from test_agent_analyze import _DraftNarrator, _ScenarioToolbox

    app = agent_gateway.create_app(toolbox=_ScenarioToolbox(), narrator=_DraftNarrator())
    with TestClient(app) as client:
        _assert_committee_in_trace(client)


def _assert_committee_in_trace(client):
    resp = client.post(
        "/api/v1/agent/analyze",
        headers=PUBLIC_HEADERS,
        json={
            "question": "现在黄金的中期趋势怎么看？",
            "risk_profile": "balanced",
            "horizon": "7d",
            "locale": "zh-CN",
        },
    )
    assert resp.status_code == 200
    analysis_id = resp.json()["analysis_id"]

    trace = client.get(
        f"/api/v1/agent/traces/{analysis_id}",
        headers={"X-API-Key": "dev-internal-key"},
    )
    assert trace.status_code == 200
    bundle = trace.json()["evidence_payload"]["bundle"]
    assert "committee" in bundle
    committee = bundle["committee"]
    assert {v["name"] for v in committee["views"]} == {
        "technical",
        "macro",
        "flow",
        "news",
    }
    assert -1.0 <= committee["fused_stance"] <= 1.0
    assert 0.0 <= committee["disagreement"] <= 1.0
    assert "regime" in bundle
