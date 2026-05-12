from __future__ import annotations

from fastapi.testclient import TestClient

import memory_service


def test_memory_search_returns_explicit_unavailable_payload_when_retriever_unavailable():
    original = memory_service.retriever
    memory_service.retriever = None
    try:
        with TestClient(memory_service.app) as client:
            resp = client.post(
                "/api/v1/memory/search",
                json={
                    "current_event_text": "CPI 高于预期",
                    "top_k": 3,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
    finally:
        memory_service.retriever = original

    assert data["storage"] in {"unavailable", "degraded"}
    assert data["status"] in {"unavailable", "degraded"}
    assert data["degraded_reason"]
    assert data["results"] == []


def test_memory_service_startup_does_not_block_on_embedding_model():
    original = memory_service.retriever
    memory_service.retriever = None
    try:
        with TestClient(memory_service.app) as client:
            health = client.get("/health")
            ready = client.get("/health/ready")
    finally:
        memory_service.retriever = original

    assert health.status_code == 200
    assert health.json()["retriever_status"] in {"not_started", "loading", "ready", "unavailable"}
    assert ready.status_code == 503
    assert "memory_retriever_not_ready" in ready.json()["errors"]
