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
