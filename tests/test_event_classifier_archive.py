"""Tests for the news event classifier and the decision-relevant news archive."""
from datetime import datetime, timedelta, timezone

import pytest

from event_classifier import classify_news
from news_archive import NewsArchive, relevance_score

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,category", [
    ("美联储宣布加息 75 个基点，鲍威尔称将继续紧缩", "monetary_policy"),
    ("Fed cuts rates by 50bp as labor market weakens", "monetary_policy"),
    ("美国 CPI 同比 6.2% 大超预期，通胀压力全面扩散", "inflation"),
    ("以色列对伊朗核设施发动空袭，中东战争风险骤升", "geopolitics"),
    ("美元指数突破 114 创二十年新高，英镑危机发酵", "usd"),
    ("世界黄金协会：央行三季度购金近 400 吨创纪录", "flows"),
    ("黄金 ETF 单周流出创两年新高", "flows"),
])
def test_classifier_categories(text, category):
    result = classify_news(text)
    assert result is not None
    assert result["category"] == category


def test_classifier_severity_grading():
    high = classify_news("美联储紧急降息，市场熔断，银行倒闭风险蔓延")
    assert high["severity"] == "high"
    medium = classify_news("美联储官员讲话提及可能调整利率路径")
    assert medium["severity"] == "medium"


def test_classifier_irrelevant_returns_none():
    assert classify_news("本地球队昨晚赢得比赛冠军") is None
    assert classify_news("新款手机今日发布，预售火爆") is None


# --------------------------------------------------------------------------- #
# Relevance scoring
# --------------------------------------------------------------------------- #
def test_relevance_scores_ordering():
    strong = relevance_score({"title": "美联储紧急加息应对通胀危机",
                              "summary": "金价剧烈波动，实际利率飙升"})
    weak = relevance_score({"title": "金店周末促销活动", "summary": "优惠多多"})
    assert strong > weak
    assert strong >= 0.5
    assert weak < 0.5


# --------------------------------------------------------------------------- #
# Archive: filter, dedupe, TTL, decayed search
# --------------------------------------------------------------------------- #
def _item(title, *, ts=None, summary=""):
    return {
        "title": title,
        "summary": summary or title,
        "published_at": (ts or NOW).isoformat(),
        "source": "test-feed",
    }


def test_archive_ingest_filters_noise(tmp_path):
    archive = NewsArchive(tmp_path / "archive.jsonl")
    report = archive.ingest([
        _item("美联储宣布加息 75 个基点，通胀创四十年新高"),
        _item("本地球队赢得比赛"),
    ], now=NOW)
    assert report["kept"] == 1
    assert report["dropped"] == 1


def test_archive_dedupes_by_normalized_title(tmp_path):
    archive = NewsArchive(tmp_path / "archive.jsonl")
    archive.ingest([_item("美联储宣布加息 75 个基点")], now=NOW)
    report = archive.ingest([_item("美联储宣布加息 75 个基点 ")], now=NOW)
    assert report["kept"] == 0
    assert report["deduped"] == 1


def test_archive_ttl_expiry(tmp_path):
    archive = NewsArchive(tmp_path / "archive.jsonl")
    old = NOW - timedelta(days=200)
    archive.ingest([_item("美联储历史性加息决议", ts=old)], now=old)
    # Expired items never come back from search...
    assert archive.search("美联储", now=NOW) == []
    # ...and prune removes them physically.
    removed = archive.prune(now=NOW)
    assert removed == 1


def test_archive_search_ranks_recent_first(tmp_path):
    archive = NewsArchive(tmp_path / "archive.jsonl")
    archive.ingest([
        _item("美联储加息决议引发金价波动", ts=NOW - timedelta(days=90)),
        _item("美联储降息预期升温支撑金价", ts=NOW - timedelta(days=2)),
    ], now=NOW)
    hits = archive.search("美联储", now=NOW)
    assert len(hits) == 2
    # Same keyword strength -> time decay must rank the recent item first.
    assert "降息预期" in hits[0]["title"]
    assert hits[0]["score"] > hits[1]["score"]
