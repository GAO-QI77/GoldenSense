"""Contract tests for the market-view and signal-ledger gateway endpoints."""
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AGENT_PUBLIC_API_KEYS", "dev-public-key")
os.environ.setdefault("AGENT_INTERNAL_API_KEYS", "dev-internal-key")

import agent_gateway  # noqa: E402
from signal_ledger import MemoryLedgerStore  # noqa: E402

PUBLIC_HEADERS = {"X-API-Key": "dev-public-key"}
INTERNAL_HEADERS = {"X-API-Key": "dev-internal-key"}


@pytest.fixture(scope="module")
def client():
    app = agent_gateway.create_app(signal_ledger_store=MemoryLedgerStore())
    with TestClient(app) as test_client:
        yield test_client


def test_market_view_requires_api_key(client):
    assert client.get("/api/v1/agent/market-view").status_code in (401, 403)


def test_market_view_contract(client):
    resp = client.get("/api/v1/agent/market-view", headers=PUBLIC_HEADERS)
    assert resp.status_code == 200
    book = resp.json()
    for horizon in ("short_term", "mid_term", "long_term"):
        section = book[horizon]
        if section["available"]:
            assert section["core_view"]
            assert section["confidence"] in {"低", "中", "高"}
            assert section["evidence"]
            assert section["invalidation"]
        else:
            assert section["degraded_reason"]
    assert book["meta"]["is_realtime"] is False


def test_signals_current_empty_ledger_is_an_explicit_success_state(client):
    resp = client.get("/api/v1/signals/current", headers=PUBLIC_HEADERS)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "empty"
    assert payload["error_code"] == "first_publication_required"
    assert payload["evidence_class"] == "live_forward"
    assert payload["next_action"] == "publish_first_weekly_signal"
    assert payload["guidance"]


def test_publish_requires_internal_key(client):
    resp = client.post("/api/v1/signals/publish", headers=PUBLIC_HEADERS)
    assert resp.status_code in (401, 403)


def test_publish_then_read_back_and_idempotent(client):
    first = client.post("/api/v1/signals/publish", headers=INTERNAL_HEADERS)
    assert first.status_code == 200
    body = first.json()
    assert body["created"] is True
    record = body["record"]
    assert record["content_hash"].startswith("sha256:")
    assert record["disclaimer"]

    again = client.post("/api/v1/signals/publish", headers=INTERNAL_HEADERS)
    assert again.status_code == 200
    assert again.json()["created"] is False
    assert again.json()["record"] == record

    current = client.get("/api/v1/signals/current", headers=PUBLIC_HEADERS)
    assert current.status_code == 200
    assert current.json() == record

    history = client.get("/api/v1/signals/history", headers=PUBLIC_HEADERS)
    assert history.status_code == 200
    assert [r["publication_id"] for r in history.json()["publications"]] == [
        record["publication_id"]
    ]

    by_id = client.get(
        f"/api/v1/signals/{record['publication_id']}", headers=PUBLIC_HEADERS
    )
    assert by_id.status_code == 200
    assert by_id.json() == record

    missing = client.get("/api/v1/signals/1999-W01", headers=PUBLIC_HEADERS)
    assert missing.status_code == 404
    assert missing.json()["detail"]["error_code"] == "publication_not_found"


def test_track_record_contract(client):
    # One publication exists (from the test above) but nothing has matured:
    # the endpoint must return an honest empty track record, never fabricate.
    resp = client.get("/api/v1/signals/track-record", headers=PUBLIC_HEADERS)
    assert resp.status_code == 200
    payload = resp.json()
    assert "per_profile" in payload and "benchmarks" in payload
    for profile in ("conservative", "balanced", "aggressive"):
        stats = payload["per_profile"][profile]
        assert "weeks_scored" in stats
        assert stats["weeks_scored"] >= 0
    assert payload["disclaimer"]
    assert payload["cost_bps"] == 5.0
