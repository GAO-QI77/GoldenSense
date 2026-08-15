"""Outcome tracking & calibration: the agent grades its own past answers.

Every analysis is persisted with a stance and a horizon. This module closes
the loop:

- ``backfill_outcomes`` joins stored analyses with realized gold prices and
  records the realized forward return for each matured horizon.
- ``calibration_summary`` converts stance vs. realized direction into hit
  rates and a Brier score, overall and per stance/confidence bucket.

The output feeds two places: the ``/api/v1/agent/calibration`` endpoint (so
the frontend can show "本系统过去 N 次偏多判断的命中率"), and -- with hard
caps -- future aggregator re-weighting. A research assistant that publishes
its own scorecard is the product's core credibility feature.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

HORIZON_DAYS = {
    "short_term": 21,
    "mid_term": 180,
    "long_term": 365,
    # Read-only compatibility for analyses persisted before the public
    # three-horizon migration.
    "24h": 1,
    "7d": 7,
    "30d": 30,
}
STANCE_TO_DIRECTION = {"偏多": 1, "偏空": -1, "中性": 0, "高风险观望": 0}
CONFIDENCE_TO_PROB = {"高": 0.75, "中": 0.65, "低": 0.55}


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


def extract_analysis_record(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a persisted trace row into a scoring record."""
    response = row.get("response_payload") or {}
    summary = response.get("summary_card") or {}
    request = row.get("request_payload") or {}
    created = _parse_ts(row.get("created_at"))
    if created is None or not summary:
        return None
    horizon = request.get("horizon") or summary.get("horizon") or "short_term"
    return {
        "analysis_id": row.get("analysis_id"),
        "created_at": created,
        "horizon": horizon if horizon in HORIZON_DAYS else "short_term",
        "stance": summary.get("stance", "中性"),
        "confidence_band": summary.get("confidence_band", "中"),
        "evidence_class": "live_forward",
    }


def realized_return(
    prices: pd.Series,
    start: datetime,
    horizon_days: int,
) -> Optional[float]:
    """Forward return from the first close at/after ``start`` over the horizon."""
    if prices.empty:
        return None
    idx = prices.index
    if idx.tz is None:
        start_naive = start.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        start_naive = start
    pos = idx.searchsorted(start_naive)
    if pos >= len(idx):
        return None
    end_ts = idx[pos] + timedelta(days=horizon_days)
    end_pos = idx.searchsorted(end_ts)
    if end_pos >= len(idx):
        return None  # horizon not matured yet
    p0 = float(prices.iloc[pos])
    p1 = float(prices.iloc[end_pos])
    if p0 <= 0:
        return None
    return p1 / p0 - 1.0


def backfill_outcomes(
    rows: List[Dict[str, Any]],
    prices: pd.Series,
) -> List[Dict[str, Any]]:
    """Attach realized forward returns to every matured analysis row."""
    outcomes: List[Dict[str, Any]] = []
    prices = prices.astype(float).dropna().sort_index()
    for row in rows:
        record = extract_analysis_record(row)
        if record is None:
            continue
        horizon_days = HORIZON_DAYS[record["horizon"]]
        realized = realized_return(prices, record["created_at"], horizon_days)
        if realized is None:
            continue
        record["realized_return"] = round(realized, 6)
        record["realized_direction"] = 1 if realized > 0 else (-1 if realized < 0 else 0)
        outcomes.append(record)
    return outcomes


def calibration_summary(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Hit rates + Brier score for directional (偏多/偏空) calls.

    Neutral / risk-gated stances are excluded from hit-rate math but counted,
    so "我们经常观望" is visible rather than hidden.
    """
    directional = [
        o for o in outcomes if STANCE_TO_DIRECTION.get(o["stance"], 0) != 0
    ]
    neutral_count = len(outcomes) - len(directional)

    summary: Dict[str, Any] = {
        "evidence_class": "live_forward",
        "total_scored": len(outcomes),
        "directional_calls": len(directional),
        "neutral_or_gated": neutral_count,
        "hit_rate": None,
        "brier_score": None,
        "by_stance": {},
        "by_confidence": {},
    }
    if not directional:
        return summary

    hits: List[float] = []
    briers: List[float] = []
    for o in directional:
        direction = STANCE_TO_DIRECTION[o["stance"]]
        hit = 1.0 if direction == o["realized_direction"] else 0.0
        prob = CONFIDENCE_TO_PROB.get(o["confidence_band"], 0.6)
        hits.append(hit)
        briers.append((prob - hit) ** 2)
        o["hit"] = bool(hit)

    summary["hit_rate"] = round(float(np.mean(hits)), 4)
    summary["brier_score"] = round(float(np.mean(briers)), 4)

    for key, group_field in (("by_stance", "stance"), ("by_confidence", "confidence_band")):
        groups: Dict[str, List[float]] = {}
        for o in directional:
            groups.setdefault(o[group_field], []).append(1.0 if o["hit"] else 0.0)
        summary[key] = {
            k: {
                "n": len(v),
                "hit_rate": round(float(np.mean(v)), 4),
                "evidence_class": "live_forward",
            }
            for k, v in groups.items()
        }
    return summary


def committee_weight_adjustments(
    summary: Dict[str, Any],
    *,
    max_shift: float = 0.20,
) -> Dict[str, float]:
    """Bounded feedback into the aggregator: overall hit-rate vs. coin flip.

    Returns a single multiplicative adjustment applied to the *fused* stance
    confidence (not to individual analysts -- per-analyst attribution needs
    more samples than a young system has). Hard-capped so online feedback can
    never run away.
    """
    hit_rate = summary.get("hit_rate")
    if hit_rate is None or summary.get("directional_calls", 0) < 20:
        return {"fused_confidence_multiplier": 1.0, "basis": "insufficient_samples"}
    edge = float(np.clip((hit_rate - 0.5) * 2.0, -1.0, 1.0))
    multiplier = float(np.clip(1.0 + edge * max_shift, 1.0 - max_shift, 1.0 + max_shift))
    return {
        "fused_confidence_multiplier": round(multiplier, 4),
        "basis": f"hit_rate={hit_rate:.2%} over {summary['directional_calls']} calls",
    }
