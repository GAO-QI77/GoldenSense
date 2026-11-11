from datetime import datetime, timezone
import json

import pytest


THROUGH = datetime(2026, 11, 10, 15, tzinfo=timezone.utc)


def _record(title, **overrides):
    return {
        "title": title,
        "summary": "The gold market reacts to CPI.",
        "source": "rss.example",
        "published_at": "2026-11-10T13:00:00Z",
        "archived_at": "2026-11-10T14:00:00Z",
        "relevance": 0.8,
        **overrides,
    }


def _archive(tmp_path, *records):
    path = tmp_path / "news.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    return path


def test_benchmark_uses_same_cutoff_and_dedupes_only_matching_text(tmp_path):
    from eval.run_news_archive_eval import evaluate_archive

    path = _archive(
        tmp_path,
        _record("Gold and CPI rise 2%", url="https://bls.gov.attacker.test/release"),
        _record("Gold and CPI rise 2%.", url="https://www.bls.gov/release", archived_at="2026-11-10T10:00:00-05:00"),
        _record("Gold and CPI rise 3%"),
        _record("Gold and CPI rise 2%", summary="Corrected regional gold market report."),
        _record("Gold revision", available_at="2026-11-10T15:00:00.001Z"),
        _record("Gold late capture", archived_at="2026-11-10T15:00:01Z"),
        _record("Gold late first seen", first_seen_at="2026-11-10T15:00:01Z"),
        _record("Gold invalid clock", first_seen_at="2026-11-10"),
        _record("Gold synthetic", source="synthetic_fallback", url="https://www.bls.gov/release"),
    )

    result = evaluate_archive(path, query="gold", through=THROUGH, limit=2)

    assert result["eligible_items"] == 4
    assert result["candidate_grouped_items"] == 1
    assert result["baseline"]["primary_url_items"] == 1
    assert result["baseline"]["same_day_text_collisions"] == 1
    assert result["candidate"]["items"] == 2
    assert result["candidate"]["primary_url_items"] == 1
    assert result["candidate"]["same_day_text_collisions"] == 0
    assert result["candidate"]["items_by_source"] == {"rss.example": 2}


def test_benchmark_keeps_first_observed_equivalent_after_source_check(tmp_path):
    from eval.run_news_archive_eval import evaluate_archive

    path = _archive(
        tmp_path,
        _record("Gold and CPI", source="second", first_seen_at="2026-11-10T14:00:00Z"),
        _record("Gold and CPI.", source="first", first_seen_at="2026-11-10T13:30:00Z"),
    )

    result = evaluate_archive(path, query="gold", through=THROUGH, limit=1)

    assert result["baseline"]["items_by_source"] == {"second": 1}
    assert result["candidate"]["items_by_source"] == {"first": 1}
    assert result["explicit_first_seen_items"] == 2


def test_benchmark_refuses_naive_cutoff(tmp_path):
    from eval.run_news_archive_eval import evaluate_archive

    with pytest.raises(ValueError, match="timezone"):
        evaluate_archive(_archive(tmp_path), query="gold", through=THROUGH.replace(tzinfo=None), limit=1)


def test_benchmark_replays_rss_archive_with_explicit_offset(tmp_path):
    from eval.run_news_archive_eval import evaluate_archive
    from news_archive import NewsArchive

    path = _archive(
        tmp_path,
        _record("Gold at cutoff", published_at="Tue, 10 Nov 2026 10:00:00 -0500"),
        _record("Gold after cutoff", published_at="Tue, 10 Nov 2026 10:00:01 -0500"),
        _record("Gold without timezone", published_at="Tue, 10 Nov 2026 14:00:00 -0000"),
    )

    result = evaluate_archive(path, query="gold", through=THROUGH)

    assert [item["title"] for item in NewsArchive(path).search("gold", now=THROUGH)] == ["Gold at cutoff"]
    assert result["legacy_rss_items"] == 1
    assert result["eligible_items"] == 1
