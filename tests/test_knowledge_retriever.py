"""Tests for unified knowledge retrieval and its gateway endpoint."""
import os
from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AGENT_PUBLIC_API_KEYS", "dev-public-key")
os.environ.setdefault("AGENT_INTERNAL_API_KEYS", "dev-internal-key")

from event_study import EventStudyLibrary  # noqa: E402
from knowledge_retriever import search_knowledge  # noqa: E402
from news_archive import NewsArchive  # noqa: E402

PUBLIC_HEADERS = {"X-API-Key": "dev-public-key"}


def _library() -> EventStudyLibrary:
    idx = pd.bdate_range("2020-01-01", periods=260)
    prices = pd.Series([1000.0 * (1.001 ** i) for i in range(len(idx))], index=idx)
    catalog = [
        {"event_id": "e1", "date": "2020-02-03", "category": "monetary_policy",
         "title": "加息决议", "summary": "s"},
        {"event_id": "e2", "date": "2020-03-02", "category": "monetary_policy",
         "title": "降息决议", "summary": "s"},
    ]
    return EventStudyLibrary(catalog=catalog, prices=prices)


def test_search_returns_analogs_and_citations(tmp_path):
    archive = NewsArchive(tmp_path / "a.jsonl")
    archive.ingest(
        [{"title": "美联储宣布加息 75 个基点", "summary": "通胀创四十年新高",
          "published_at": datetime.now(timezone.utc).isoformat(), "source": "t"}],
    )
    result = search_knowledge(
        "美联储加息对黄金的影响", event_library=_library(), news_archive=archive
    )
    assert result["classification"]["category"] == "monetary_policy"
    assert result["event_analogs"]["n_30d"] == 2
    assert result["news_hits"]
    types = {c["type"] for c in result["citations"]}
    assert types == {"event_study", "news_archive"}
    for citation in result["citations"]:
        assert citation["title"] and citation["date"]


def test_unclassifiable_query_degrades_explicitly(tmp_path):
    result = search_knowledge(
        "今天天气如何",
        event_library=_library(),
        news_archive=NewsArchive(tmp_path / "b.jsonl"),
    )
    assert result["event_analogs"] is None
    assert result["degraded"]["event_analogs"] == "query_not_classifiable"


def test_knowledge_endpoint_contract():
    import agent_gateway
    from signal_ledger import MemoryLedgerStore

    app = agent_gateway.create_app(signal_ledger_store=MemoryLedgerStore())
    with TestClient(app) as client:
        assert client.get(
            "/api/v1/agent/knowledge/search", params={"q": "美联储加息"}
        ).status_code in (401, 403)

        resp = client.get(
            "/api/v1/agent/knowledge/search",
            params={"q": "美联储加息后黄金历史表现"},
            headers=PUBLIC_HEADERS,
        )
        assert resp.status_code == 200
        payload = resp.json()
        for key in ("query", "classification", "event_analogs", "news_hits",
                    "citations", "degraded"):
            assert key in payload
        # Real catalog: monetary_policy analogs must be present with n>=10.
        assert payload["classification"]["category"] == "monetary_policy"
        assert payload["event_analogs"]["n_30d"] >= 10
