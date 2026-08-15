from __future__ import annotations

import json

from fastapi.testclient import TestClient

from agent_gateway import create_app
from horizon_contracts import (
    PUBLIC_HORIZONS,
    public_horizon_payload,
    to_legacy_quant_horizon,
)

from test_agent_analyze import _ScenarioToolbox


def test_public_horizons_are_research_periods_only() -> None:
    assert PUBLIC_HORIZONS == ("short_term", "mid_term", "long_term")


def test_legacy_adapter_is_one_way() -> None:
    assert to_legacy_quant_horizon("short_term") == "T+1"
    assert to_legacy_quant_horizon("mid_term") == "T+7"
    assert to_legacy_quant_horizon("long_term") == "T+30"
    assert public_horizon_payload("T+1") == "short_term"
    assert public_horizon_payload("T+7") == "mid_term"
    assert public_horizon_payload("T+30") == "long_term"


def test_public_forecast_response_never_leaks_legacy_horizon_names() -> None:
    app = create_app(toolbox=_ScenarioToolbox())
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/agent/forecasts/current",
            headers={"X-API-Key": "dev-public-key"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert {item["horizon"] for item in payload["horizon_forecasts"]} == set(PUBLIC_HORIZONS)
    serialized = json.dumps(payload, ensure_ascii=False)
    for legacy in ('"24h"', '"7d"', '"30d"', '"T+1"', '"T+7"', '"T+30"'):
        assert legacy not in serialized
