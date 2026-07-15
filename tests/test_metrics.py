"""Tests for the in-process metrics registry and the /metrics endpoint."""
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AGENT_PUBLIC_API_KEYS", "dev-public-key")
os.environ.setdefault("AGENT_INTERNAL_API_KEYS", "dev-internal-key")

import agent_gateway  # noqa: E402
from service_metrics import MetricsRegistry  # noqa: E402


def test_registry_records_latency_and_error_rate():
    reg = MetricsRegistry()
    reg.record_request("/x", status_code=200, elapsed_ms=10.0)
    reg.record_request("/x", status_code=200, elapsed_ms=30.0)
    reg.record_request("/x", status_code=503, elapsed_ms=5.0)
    snap = reg.snapshot()
    route = snap["routes"]["/x"]
    assert route["count"] == 3
    assert route["errors"] == 1
    assert route["error_rate"] == round(1 / 3, 4)
    assert route["latency_ms_avg"] == 15.0
    assert route["latency_ms_max"] == 30.0
    assert snap["status_classes"]["2xx"] == 2
    assert snap["status_classes"]["5xx"] == 1


def test_registry_domain_counters():
    reg = MetricsRegistry()
    reg.incr("analyze_total")
    reg.incr("degradation:news_degraded", 2)
    snap = reg.snapshot()
    assert snap["domain_counters"]["analyze_total"] == 1
    assert snap["domain_counters"]["degradation:news_degraded"] == 2


@pytest.fixture(scope="module")
def client():
    app = agent_gateway.create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_metrics_endpoint_requires_internal_key(client):
    # Public key must not reach the internal metrics endpoint.
    resp = client.get("/metrics", headers={"X-API-Key": "dev-public-key"})
    assert resp.status_code in (401, 403)


def test_metrics_endpoint_returns_snapshot_and_records_traffic(client):
    # Generate some traffic first.
    client.get("/health", headers={"X-API-Key": "dev-public-key"})
    resp = client.get("/metrics", headers={"X-API-Key": "dev-internal-key"})
    assert resp.status_code == 200
    snap = resp.json()
    assert "routes" in snap and "status_classes" in snap
    assert "uptime_seconds" in snap
    # The /health route should have been recorded by the middleware.
    assert any("/health" in route for route in snap["routes"])
