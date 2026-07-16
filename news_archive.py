"""Decision-relevant news archive: filtered in, decayed out.

The decaying layer of the knowledge base. Incoming news items are kept only
when they carry decision-relevant information for gold research (classifiable
macro category and/or strong market terms); noise is dropped at the door.
Stored items expire after ``TTL_DAYS`` (stale news is worse than no news) and
search ranks hits by keyword match strength multiplied by exponential time
decay -- so "过去 90 天关于央行购金的报道" surfaces the recent and relevant,
never the ancient.

Storage is a local JSONL file (offline/CI friendly). An embedding/pgvector
backend is a drop-in upgrade path behind the same class interface.
"""
from __future__ import annotations

import json
import math
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from event_classifier import HIGH_SEVERITY_TERMS, classify_news

TTL_DAYS = 180
DECISION_THRESHOLD = 0.5
DECAY_HALF_LIFE_DAYS = 30.0

_WS_RE = re.compile(r"\s+")
_CJK_RUN_RE = re.compile(r"[一-鿿]+")


def _query_terms(query: str) -> List[str]:
    """Whitespace tokens for latin text; overlapping bigrams for CJK runs
    (Chinese queries carry no whitespace, so substring matching needs
    segmentation)."""
    terms: List[str] = []
    for token in _WS_RE.split((query or "").strip().lower()):
        if not token:
            continue
        cjk_runs = _CJK_RUN_RE.findall(token)
        latin = _CJK_RUN_RE.sub(" ", token).strip()
        if latin:
            terms.extend(t for t in latin.split() if t)
        for run in cjk_runs:
            if len(run) == 1:
                terms.append(run)
            else:
                terms.extend(run[i:i + 2] for i in range(len(run) - 1))
    return terms


def _normalize_title(title: str) -> str:
    return _WS_RE.sub("", (title or "").strip().lower())


def _parse_ts(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def relevance_score(item: Dict[str, Any]) -> float:
    """0..1 heuristic: classifiable category + strong terms => relevant."""
    text = f"{item.get('title', '')} {item.get('summary', '')}"
    score = 0.0
    classified = classify_news(text)
    if classified:
        score += 0.5
        if classified["severity"] == "high":
            score += 0.2
        score += min(0.2, 0.05 * len(classified["matched"]))
    lower = text.lower()
    if any(term in lower for term in ("黄金", "金价", "gold", "实际利率", "避险")):
        score += 0.1
    return round(min(score, 1.0), 4)


class NewsArchive:
    """Append-mostly JSONL archive with TTL pruning and decayed search."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _load(self) -> List[Dict[str, Any]]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def _write_all(self, records: List[Dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _is_expired(record: Dict[str, Any], now: datetime) -> bool:
        ts = _parse_ts(record.get("published_at"))
        if ts is None:
            return True
        return (now - ts).days > TTL_DAYS

    # ------------------------------------------------------------------ #
    def ingest(
        self,
        items: List[Dict[str, Any]],
        *,
        now: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """Filter for decision relevance, dedupe by title, append."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            existing = self._load()
            seen = {_normalize_title(r.get("title", "")) for r in existing}
            kept = dropped = deduped = 0
            for item in items:
                key = _normalize_title(item.get("title", ""))
                if not key:
                    dropped += 1
                    continue
                if key in seen:
                    deduped += 1
                    continue
                score = relevance_score(item)
                if score < DECISION_THRESHOLD:
                    dropped += 1
                    continue
                text = f"{item.get('title', '')} {item.get('summary', '')}"
                record = {
                    "title": item.get("title"),
                    "summary": item.get("summary"),
                    "source": item.get("source"),
                    "published_at": item.get("published_at") or now.isoformat(),
                    "relevance": score,
                    "classification": classify_news(text),
                    "archived_at": now.isoformat(),
                }
                existing.append(record)
                seen.add(key)
                kept += 1
            if kept:
                self._write_all(existing)
        return {"kept": kept, "dropped": dropped, "deduped": deduped}

    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        *,
        now: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Keyword hits x exponential time decay; expired items excluded."""
        now = now or datetime.now(timezone.utc)
        terms = _query_terms(query)
        if not terms:
            return []
        results: List[Dict[str, Any]] = []
        for record in self._load():
            if self._is_expired(record, now):
                continue
            text = f"{record.get('title', '')} {record.get('summary', '')}".lower()
            # Normalized by term count so long queries don't inflate scores.
            match_strength = sum(1.0 for t in terms if t in text) / len(terms)
            if match_strength == 0:
                continue
            ts = _parse_ts(record.get("published_at")) or now
            age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            decay = math.exp(-math.log(2) * age_days / DECAY_HALF_LIFE_DAYS)
            hit = dict(record)
            hit["score"] = round(match_strength * decay * float(record.get("relevance", 0.5)), 6)
            results.append(hit)
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def prune(self, *, now: Optional[datetime] = None) -> int:
        """Physically remove expired records; returns how many were removed."""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            records = self._load()
            fresh = [r for r in records if not self._is_expired(r, now)]
            removed = len(records) - len(fresh)
            if removed:
                self._write_all(fresh)
        return removed
