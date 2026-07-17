"""Contract tests for the digest subscription endpoints."""
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AGENT_PUBLIC_API_KEYS", "dev-public-key")
os.environ.setdefault("AGENT_INTERNAL_API_KEYS", "dev-internal-key")

import agent_gateway  # noqa: E402
from signal_ledger import MemoryLedgerStore  # noqa: E402
from subscriptions import SubscriptionStore  # noqa: E402

PUBLIC_HEADERS = {"X-API-Key": "dev-public-key"}


@pytest.fixture()
def client(tmp_path):
    app = agent_gateway.create_app(
        signal_ledger_store=MemoryLedgerStore(),
        subscription_store=SubscriptionStore(tmp_path / "subs.jsonl"),
    )
    with TestClient(app) as test_client:
        yield test_client


def test_subscribe_requires_api_key(client):
    assert client.post(
        "/api/v1/subscriptions", json={"email": "a@example.com"}
    ).status_code in (401, 403)


def test_subscribe_masks_email_and_never_leaks_token(client):
    resp = client.post(
        "/api/v1/subscriptions", headers=PUBLIC_HEADERS,
        json={"email": "User@Example.com"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload == {"email_masked": "u***@example.com", "created": True}

    again = client.post(
        "/api/v1/subscriptions", headers=PUBLIC_HEADERS,
        json={"email": "user@example.com"},
    ).json()
    assert again["created"] is False


def test_subscribe_invalid_email_422(client):
    resp = client.post(
        "/api/v1/subscriptions", headers=PUBLIC_HEADERS,
        json={"email": "not-an-email"},
    )
    assert resp.status_code == 422


def test_unsubscribe_link_requires_no_key(tmp_path):
    store = SubscriptionStore(tmp_path / "subs.jsonl")
    sub = store.subscribe("user@example.com")
    app = agent_gateway.create_app(
        signal_ledger_store=MemoryLedgerStore(), subscription_store=store,
    )
    with TestClient(app) as client:
        resp = client.get(f"/api/v1/subscriptions/unsubscribe?token={sub['token']}")
        assert resp.status_code == 200
        assert resp.json()["unsubscribed"] is True
        # Idempotent second click.
        assert client.get(
            f"/api/v1/subscriptions/unsubscribe?token={sub['token']}"
        ).json()["unsubscribed"] is False
    assert store.active_subscribers() == []
