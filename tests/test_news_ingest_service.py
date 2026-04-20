from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from news_ingest_service import NewsIngestConfig, create_app, normalize_news_items


class _FakeNewsLoader:
    def fetch_news(self):
        return [
            {
                "title": "Fed signals dovish turn",
                "summary": "Officials open the door to lower rates.",
                "published": datetime.now(timezone.utc).isoformat(),
                "source": "reuters.com",
            },
            {
                "title": "Fed signals dovish turn",
                "summary": "Duplicate item should be removed.",
                "published": datetime.now(timezone.utc).isoformat(),
                "source": "duplicate.com",
            },
        ]

    def analyze_causality(self, news_items):
        scored = []
        for item in news_items:
            copy = dict(item)
            copy.update({"total": 1.4, "rates": 1.0, "inflation": 0.0, "risk": 0.0, "fx": 0.0})
            scored.append(copy)
        return scored


class _FailingNewsLoader:
    def fetch_news(self):
        raise RuntimeError("rss_down")

    def analyze_causality(self, news_items):
        return []


class _MemoryPersistence:
    def __init__(self, payload=None):
        self.payload = payload

    def save(self, response):
        self.payload = response

    def load(self):
        return self.payload


def test_normalize_news_items_dedupes_and_scores():
    payload = normalize_news_items(_FakeNewsLoader().analyze_causality(_FakeNewsLoader().fetch_news()))
    assert len(payload.items) == 1
    assert payload.items[0].sentiment_score > 0
    assert "rates" in payload.items[0].categories


def test_recent_news_endpoint_contract():
    app = create_app(news_loader=_FakeNewsLoader(), start_background_task=False)
    with TestClient(app) as client:
        refresh = client.post("/api/v1/news/refresh")
        assert refresh.status_code == 200

        resp = client.get("/api/v1/news/recent", params={"limit": 5, "q": "dovish"})
        assert resp.status_code == 200
        data = resp.json()
    assert "items" in data
    assert data["status"] == "ok"
    assert len(data["items"]) == 1
    assert data["items"][0]["source"] == "reuters.com"


def test_recent_news_refresh_uses_sample_fallback_when_upstream_fails():
    app = create_app(news_loader=_FailingNewsLoader(), start_background_task=False)
    with TestClient(app) as client:
        resp = client.post("/api/v1/news/refresh")
        assert resp.status_code == 200
        data = resp.json()
    assert len(data["items"]) >= 1
    assert data["status"] == "degraded"
    assert data["degraded_reason"]
    assert data["items"][0]["source"] == "synthetic_fallback"


def test_recent_news_refresh_prefers_cached_payload_before_sample_fallback():
    cached = normalize_news_items(_FakeNewsLoader().analyze_causality(_FakeNewsLoader().fetch_news()))
    app = create_app(
        news_loader=_FailingNewsLoader(),
        persistence=_MemoryPersistence(cached),
        config=NewsIngestConfig(
            allow_sample_fallback=True,
            stale_cache_grace_seconds=3600,
        ),
        start_background_task=False,
    )
    with TestClient(app) as client:
        resp = client.post("/api/v1/news/refresh")
        assert resp.status_code == 200
        data = resp.json()
    assert len(data["items"]) == 1
    assert data["status"] == "degraded"
    assert data["degraded_reason"].startswith("cache_fallback:")
    assert data["items"][0]["source"] == "reuters.com"
