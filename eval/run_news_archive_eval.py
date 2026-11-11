"""Compare archive search and a source-aware candidate without changing live results.

Run from the repository root with python -m eval.run_news_archive_eval.
Reports measure ranking and archive coverage. They have no independent fact labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

from news_archive import NewsArchive, parse_archive_timestamp
from news_provenance import primary_authority_for_url


def _timestamp(value: str) -> datetime:
    timestamp = parse_archive_timestamp(value, require_timezone=True)
    if timestamp is None:
        raise ValueError("news timestamps require a valid time and explicit timezone")
    return timestamp.astimezone(timezone.utc)


def _uses_rss_clock(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return True
    return False


def _text_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(".!?。！？ ")


def _metrics(items: list[dict]) -> dict:
    return {
        "items": len(items),
        "same_day_text_collisions": len(items) - len({item["dedup_key"] for item in items}),
        "primary_url_items": sum(item["primary"] for item in items),
        "items_by_source": dict(sorted(Counter(item["source"] for item in items).items())),
    }


def evaluate_archive(path: Path, *, query: str, through: datetime, limit: int = 10) -> dict:
    if through.utcoffset() is None:
        raise ValueError("through requires an explicit timezone")
    if limit < 1 or not query.strip():
        raise ValueError("a nonempty query and a positive limit are required")
    if not path.is_file():
        raise FileNotFoundError(path)

    matched = NewsArchive(path).search(query, now=through, limit=sys.maxsize)
    baseline = []
    for record in matched:
        if record.get("source") == "synthetic_fallback":
            continue
        try:
            published = _timestamp(record["published_at"])
            archived = _timestamp(record["archived_at"])
            clocks = {
                field: _timestamp(record[field])
                for field in ("first_seen_at", "updated", "updated_at", "available_at")
                if record.get(field) is not None
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if max(published, archived, *clocks.values()) > through:
            continue
        url = record.get("url")
        url = url if isinstance(url, str) and url.strip() else None
        baseline.append({
            "score": record["score"],
            "source": record.get("source") or "unknown",
            "has_url": url is not None,
            "primary": primary_authority_for_url(url) is not None,
            "explicit_first_seen": "first_seen_at" in clocks,
            "legacy_rss": any(_uses_rss_clock(record[field]) for field in ("published_at", "archived_at")),
            "observed_at": max(published, clocks.get("first_seen_at", archived)),
            "dedup_key": (published.date(), _text_key(record["title"]), _text_key(record.get("summary") or "")),
        })

    # Prefer the primary-source hostname in each group. Within a source tier,
    # keep the first observed copy, using archive time as a proxy.
    groups = {}
    for item in sorted(baseline, key=lambda item: (not item["primary"], item["observed_at"], -item["score"])):
        groups.setdefault(item["dedup_key"], item)
    candidate = sorted(
        groups.values(), key=lambda item: (not item["primary"], -item["score"], item["observed_at"]),
    )
    baseline_top = baseline[:limit]
    candidate_top = candidate[:limit]
    return {
        "protocol": "news_archive_source_order_v1",
        "evidence_status": "archive_diagnostics_only",
        "through": through.astimezone(timezone.utc).isoformat(),
        "query": query,
        "limit": limit,
        "eligible_items": len(baseline),
        "legacy_rss_items": sum(item["legacy_rss"] for item in baseline),
        "items_with_url": sum(item["has_url"] for item in baseline),
        "explicit_first_seen_items": sum(item["explicit_first_seen"] for item in baseline),
        "candidate_grouped_items": len(baseline) - len(candidate),
        "top_order_changed": baseline_top != candidate_top,
        "baseline": _metrics(baseline_top),
        "candidate": _metrics(candidate_top),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Existing news_archive.jsonl")
    parser.add_argument("--through", required=True, help="Cutoff with an explicit UTC offset")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", default=10, type=int)
    args = parser.parse_args()
    try:
        original = args.archive.read_bytes()
        report = evaluate_archive(args.archive, query=args.query, through=_timestamp(args.through), limit=args.limit)
        if args.archive.read_bytes() != original:
            raise ValueError("archive changed during evaluation")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report["archive_sha256"] = hashlib.sha256(original).hexdigest()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
