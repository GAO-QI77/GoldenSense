"""Tests for GET /api/v1/agent/event-alert (high-severity news banner feed)."""
import os
from datetime import datetime, timezone

from fastapi.testclient import TestClient

os.environ.setdefault("AGENT_PUBLIC_API_KEYS", "dev-public-key")
os.environ.setdefault("AGENT_INTERNAL_API_KEYS", "dev-internal-key")

import agent_gateway  # noqa: E402
from service_contracts import NewsEventItem, RecentNewsResponse  # noqa: E402
from signal_ledger import MemoryLedgerStore  # noqa: E402

PUBLIC_HEADERS = {"X-API-Key": "dev-public-key"}


def _news_item(title: str, summary: str = "") -> NewsEventItem:
    return NewsEventItem(
        event_id=f"e-{abs(hash(title)) % 10_000}",
        published_at=datetime.now(timezone.utc),
        title=title,
        summary=summary or title,
        source="test-feed",
        normalized_event=title,
        sentiment_score=0.0,
        importance=1.0,
        categories=["macro"],
    )


class _NewsToolbox:
    def __init__(self, titles):
        self._titles = titles
        self.calls = 0

    async def search_recent_news(self, query, limit=6):
        self.calls += 1
        return RecentNewsResponse(
            as_of=datetime.now(timezone.utc),
            freshness_seconds=10,
            status="ok",
            items=[_news_item(t) for t in self._titles],
        )


class _BrokenToolbox:
    async def search_recent_news(self, query, limit=6):
        raise RuntimeError("news service down")


def _client(toolbox) -> TestClient:
    app = agent_gateway.create_app(
        toolbox=toolbox, signal_ledger_store=MemoryLedgerStore()
    )
    return TestClient(app)


def test_requires_api_key():
    with _client(_NewsToolbox([])) as client:
        assert client.get("/api/v1/agent/event-alert").status_code in (401, 403)


def test_high_severity_news_activates_alert():
    toolbox = _NewsToolbox([
        "以色列空袭伊朗核设施，中东战争风险骤升",
        "金店周末促销",
    ])
    with _client(toolbox) as client:
        payload = client.get("/api/v1/agent/event-alert", headers=PUBLIC_HEADERS).json()
        assert payload["active"] is True
        assert payload["severity"] == "high"
        assert payload["category"] == "geopolitics"
        assert "空袭" in payload["title"]
        assert payload["checked_at"]


def test_ordinary_news_stays_inactive():
    toolbox = _NewsToolbox(["美联储官员讲话提及可能调整利率路径"])
    with _client(toolbox) as client:
        payload = client.get("/api/v1/agent/event-alert", headers=PUBLIC_HEADERS).json()
        assert payload["active"] is False


def test_toolbox_failure_degrades_to_inactive():
    with _client(_BrokenToolbox()) as client:
        payload = client.get("/api/v1/agent/event-alert", headers=PUBLIC_HEADERS).json()
        assert payload["active"] is False
        assert "RuntimeError" in payload["degraded"]


def test_ttl_cache_prevents_repeated_toolbox_calls():
    toolbox = _NewsToolbox(["美联储紧急降息，市场熔断"])
    with _client(toolbox) as client:
        first = client.get("/api/v1/agent/event-alert", headers=PUBLIC_HEADERS).json()
        second = client.get("/api/v1/agent/event-alert", headers=PUBLIC_HEADERS).json()
        assert first == second
        assert toolbox.calls == 1
