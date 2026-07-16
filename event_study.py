"""Event-study library: curated macro events scored on real forward returns.

The permanent layer of the knowledge base. ``knowledge/events_catalog.jsonl``
holds hand-curated macro events (Fed pivots, CPI surprises, geopolitical
shocks, dollar extremes, flow regimes). This module joins them with the long
gold price history and computes, per event, the *actual* 5/30/90-day forward
returns -- so an agent citing "历史上 n 次类似事件后 30 天平均 +x%" is quoting
an auditable event study, not a vibe. The catalog stores no return numbers:
every statistic is recomputed from price data, never hand-written.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

CATEGORIES = ("monetary_policy", "inflation", "geopolitics", "usd", "flows")
CATALOG_PATH = Path(__file__).resolve().parent / "knowledge" / "events_catalog.jsonl"
HORIZONS_BDAYS = {"fwd_5d": 5, "fwd_30d": 30, "fwd_90d": 90}
_TTL_SECONDS = 6 * 3600


def load_catalog(path: Path | str = CATALOG_PATH) -> List[Dict[str, Any]]:
    """Load and validate the curated catalog; malformed entries raise."""
    events: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        missing = {"event_id", "date", "category", "title", "summary"} - set(event)
        if missing:
            raise ValueError(f"catalog entry missing fields {missing}: {event}")
        if event["category"] not in CATEGORIES:
            raise ValueError(f"unknown category {event['category']!r} in {event['event_id']}")
        pd.Timestamp(event["date"])  # raises on bad dates
        events.append(event)
    return events


def _forward_return(prices: pd.Series, anchor_pos: int, bdays: int) -> Optional[float]:
    end_pos = anchor_pos + bdays
    if end_pos >= len(prices):
        return None
    p0 = float(prices.iloc[anchor_pos])
    if p0 <= 0:
        return None
    return float(prices.iloc[end_pos]) / p0 - 1.0


def compute_event_study(
    catalog: List[Dict[str, Any]],
    prices: pd.Series,
) -> Dict[str, Any]:
    """Per-event forward returns + per-category aggregates.

    The anchor is the first trading close at/after the event date; events
    before the data start or without enough forward history score None and
    are excluded from aggregates (honest sample accounting, no padding).
    """
    prices = prices.astype(float).dropna().sort_index()
    scored: List[Dict[str, Any]] = []
    for event in catalog:
        when = pd.Timestamp(event["date"])
        pos = int(prices.index.searchsorted(when))
        row: Dict[str, Any] = {
            "event_id": event["event_id"],
            "date": event["date"],
            "category": event["category"],
            "title": event["title"],
        }
        # searchsorted returns 0 both for "before data" and "first day";
        # only anchor if the event date is actually inside the data span.
        in_span = pos < len(prices) and when >= prices.index[0]
        for key, bdays in HORIZONS_BDAYS.items():
            row[key] = _forward_return(prices, pos, bdays) if in_span else None
        scored.append(row)

    by_category: Dict[str, Dict[str, Any]] = {}
    for category in CATEGORIES:
        rows = [r for r in scored if r["category"] == category]
        agg: Dict[str, Any] = {"n_events": len(rows)}
        for key in HORIZONS_BDAYS:
            values = [r[key] for r in rows if r[key] is not None]
            suffix = key.replace("fwd_", "")
            agg[f"n_{suffix}"] = len(values)
            if values:
                arr = np.array(values)
                agg[f"mean_{suffix}"] = round(float(arr.mean()), 6)
                agg[f"median_{suffix}"] = round(float(np.median(arr)), 6)
                agg[f"positive_share_{suffix}"] = round(float((arr > 0).mean()), 4)
            else:
                agg[f"mean_{suffix}"] = None
                agg[f"median_{suffix}"] = None
                agg[f"positive_share_{suffix}"] = None
        by_category[category] = agg

    return {"events": scored, "by_category": by_category}


class EventStudyLibrary:
    """TTL-cached event study over the repo-local long dataset."""

    def __init__(
        self,
        *,
        catalog: Optional[List[Dict[str, Any]]] = None,
        prices: Optional[pd.Series] = None,
        ttl_seconds: float = _TTL_SECONDS,
    ):
        self._catalog = catalog
        self._prices = prices
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._study: Optional[Dict[str, Any]] = None
        self._computed_at = 0.0

    def _get_study(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._study is not None and (time.time() - self._computed_at) < self._ttl:
                return self._study
            try:
                catalog = self._catalog if self._catalog is not None else load_catalog()
                prices = self._prices
                if prices is None:
                    from data_sources import load_market_data

                    raw, _source = load_market_data()
                    prices = raw["Gold"]
                self._study = compute_event_study(catalog, prices)
                self._computed_at = time.time()
            except Exception:
                self._study = None
            return self._study

    def analogs_for(self, category: Optional[str]) -> Optional[Dict[str, Any]]:
        """Aggregate stats + recent event citations for one category."""
        if category not in CATEGORIES:
            return None
        study = self._get_study()
        if study is None:
            return None
        agg = dict(study["by_category"].get(category) or {})
        if not agg or not agg.get("n_events"):
            return None
        recent = [r for r in study["events"] if r["category"] == category][-5:]
        agg["category"] = category
        agg["recent_events"] = [
            {"event_id": r["event_id"], "date": r["date"], "title": r["title"],
             "fwd_30d": r["fwd_30d"]}
            for r in reversed(recent)
        ]
        return agg


# Process-wide singleton (mirrors research_context.shared_context).
shared_event_library = EventStudyLibrary()
