"""Unified knowledge retrieval: event-study analogs + archived news, cited.

One entry point for both agents. A query is classified onto the event
taxonomy; if it lands, the event-study library supplies the historical-analog
statistics (real computed forward returns, with event citations); the news
archive supplies recent decision-relevant coverage. Every hit carries a
citation and both sides degrade explicitly and independently.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from event_classifier import classify_news
from event_study import EventStudyLibrary, shared_event_library
from news_archive import NewsArchive

DEFAULT_ARCHIVE_PATH = "data_cache/news_archive.jsonl"


def search_knowledge(
    query: str,
    *,
    event_library: Optional[EventStudyLibrary] = None,
    news_archive: Optional[NewsArchive] = None,
    limit: int = 8,
) -> Dict[str, Any]:
    event_library = event_library or shared_event_library
    news_archive = news_archive or NewsArchive(DEFAULT_ARCHIVE_PATH)

    degraded: Dict[str, str] = {}
    classification = classify_news(query)

    event_analogs: Optional[Dict[str, Any]] = None
    if classification:
        try:
            event_analogs = event_library.analogs_for(classification["category"])
            if event_analogs is None:
                degraded["event_analogs"] = "event_study_unavailable"
        except Exception as exc:
            degraded["event_analogs"] = f"{type(exc).__name__}: {exc}"
    else:
        degraded["event_analogs"] = "query_not_classifiable"

    news_hits = []
    try:
        news_hits = news_archive.search(query, limit=limit)
    except Exception as exc:
        degraded["news_hits"] = f"{type(exc).__name__}: {exc}"

    citations = []
    if event_analogs:
        for ref in event_analogs.get("recent_events", []):
            citations.append({
                "type": "event_study",
                "id": ref["event_id"],
                "date": ref["date"],
                "title": ref["title"],
            })
    for hit in news_hits:
        citations.append({
            "type": "news_archive",
            "id": None,
            "date": hit.get("published_at"),
            "title": hit.get("title"),
            "source": hit.get("source"),
        })

    return {
        "query": query,
        "classification": classification,
        "event_analogs": event_analogs,
        "news_hits": news_hits,
        "citations": citations,
        "degraded": degraded,
    }
