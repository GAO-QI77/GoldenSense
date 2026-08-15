"""Immutable weekly signal ledger + forward shadow-portfolio scoring.

The accountability layer of the research product. Once a week the system
freezes what it said -- the per-profile reference allocation ranges, a
one-line market view per horizon, and the evidence snapshot behind them --
into an append-only ledger. Records are:

- **idempotent per ISO week**: republishing within the same week returns the
  existing record byte-for-byte; nothing is ever overwritten;
- **tamper-evident**: each record carries a sha256 content hash over its
  canonical JSON;
- **honest from day one**: the track record starts empty and accrues forward.
  There is no backfilled history -- backtests live elsewhere and are labeled
  as backtests.

Scoring holds, for each risk profile, a hypothetical shadow portfolio at the
published midpoint gold weight (rest in zero-yield cash), rebalanced at each
publication, net of turnover costs, and only over *matured* weeks (the next
publication's rebalance price must exist).
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import pandas as pd

from allocation import ALLOCATION_DISCLAIMER
from market_view import build_market_view, summarize_for_ledger

TRACK_RECORD_DISCLAIMER = (
    "以下为假设性影子组合的前向追踪记录（发布制、含换手成本、非真实账户），"
    "自首次发布起逐周累积，不含任何回填历史；不构成投资建议。"
)

PROFILES = ("conservative", "balanced", "aggressive")
WEEKS_PER_YEAR = 52


# --------------------------------------------------------------------------- #
# Stores
# --------------------------------------------------------------------------- #
class LedgerStore(Protocol):
    def append(self, record: Dict[str, Any]) -> None: ...
    def load_all(self) -> List[Dict[str, Any]]: ...


class MemoryLedgerStore:
    """In-memory store for tests and ephemeral environments."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._records.append(json.loads(json.dumps(record, ensure_ascii=False)))

    def load_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._records]


class JsonlLedgerStore:
    """Append-only JSONL file store; existing lines are never rewritten."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def load_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            if not self._path.exists():
                return []
            records = []
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
            return records


# --------------------------------------------------------------------------- #
# Publication
# --------------------------------------------------------------------------- #
def publication_id_for(when: datetime) -> str:
    iso = when.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _canonical_hash(record: Dict[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "content_hash"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_publication(record: Dict[str, Any]) -> bool:
    return record.get("content_hash") == _canonical_hash(record)


def build_publication(ctx: Dict[str, Any], *, now: datetime) -> Dict[str, Any]:
    """Freeze one weekly publication from a research-context dict.

    Every number is read from already-validated context blocks; nothing is
    recomputed here.
    """
    allocations: Dict[str, Any] = {}
    for profile in PROFILES:
        adv = (ctx.get("allocation") or {}).get(profile) or {}
        rng = adv.get("reference_range_pct") or adv.get("recommended_range_pct")
        if rng and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
            allocations[profile] = {
                "range_pct": [lo, hi],
                "midpoint": round((lo + hi) / 2.0, 2),
            }
        else:
            allocations[profile] = {"range_pct": None, "midpoint": None}

    book = build_market_view(ctx)
    regime_latest = ((ctx.get("regime_posterior") or {}).get("latest")) or None
    fair = ctx.get("fair_value") or {}
    factors = ctx.get("macro_factors") or {}

    record: Dict[str, Any] = {
        "evidence_class": "live_forward",
        "publication_id": publication_id_for(now),
        "published_at": (
            now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        ).astimezone(timezone.utc).isoformat(),
        "data_asof": ctx.get("data_asof"),
        "data_source": ctx.get("data_source"),
        "allocations": allocations,
        "market_view_summary": summarize_for_ledger(book),
        "evidence_snapshot": {
            "regime_posterior": regime_latest,
            "fair_value_deviation_pct": fair.get("deviation_pct"),
            "fair_value_deviation_z": fair.get("deviation_z"),
            "macro_composite": factors.get("composite"),
        },
        "degraded": dict(ctx.get("degraded", {})),
        "disclaimer": ALLOCATION_DISCLAIMER,
    }
    record["content_hash"] = _canonical_hash(record)
    return record


def publish_weekly(
    store: LedgerStore,
    ctx: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Publish this ISO week's record once; republishing returns the original."""
    now = now or datetime.now(timezone.utc)
    pub_id = publication_id_for(now)
    for existing in store.load_all():
        if existing.get("publication_id") == pub_id:
            return existing, False
    record = build_publication(ctx, now=now)
    store.append(record)
    return record, True


# --------------------------------------------------------------------------- #
# Track record scoring
# --------------------------------------------------------------------------- #
def _rebalance_price(prices: pd.Series, published_at: str) -> Optional[Tuple[pd.Timestamp, float]]:
    """First close at/after the publication date; None if not yet available."""
    try:
        when = pd.Timestamp(published_at)
    except (ValueError, TypeError):
        return None
    if when.tzinfo is not None:
        when = when.tz_convert("UTC").tz_localize(None)
    when = when.normalize()
    pos = prices.index.searchsorted(when)
    if pos >= len(prices):
        return None
    return prices.index[pos], float(prices.iloc[pos])


def _profile_stats(weekly_returns: List[float]) -> Dict[str, Any]:
    if not weekly_returns:
        return {
            "weeks_scored": 0,
            "cum_return": None,
            "ann_vol": None,
            "max_drawdown": None,
            "sharpe": None,
        }
    import numpy as np

    equity = np.cumprod([1.0] + [1.0 + r for r in weekly_returns])
    cum_return = float(equity[-1] - 1.0)
    running_max = np.maximum.accumulate(equity)
    max_dd = float((equity / running_max - 1.0).min())
    if len(weekly_returns) >= 2:
        vol = float(np.std(weekly_returns, ddof=1) * np.sqrt(WEEKS_PER_YEAR))
        mean_ann = float(np.mean(weekly_returns) * WEEKS_PER_YEAR)
        sharpe = round(mean_ann / vol, 3) if vol > 0 else None
        ann_vol = round(vol, 4)
    else:
        ann_vol = None
        sharpe = None
    return {
        "weeks_scored": len(weekly_returns),
        "cum_return": round(cum_return, 6),
        "ann_vol": ann_vol,
        "max_drawdown": round(max_dd, 6),
        "sharpe": sharpe,
    }


def score_track_record(
    publications: List[Dict[str, Any]],
    prices: pd.Series,
    *,
    cost_bps: float = 5.0,
) -> Dict[str, Any]:
    """Score matured weeks of the shadow portfolios against fixed benchmarks."""
    prices = prices.astype(float).dropna().sort_index()
    ordered = sorted(publications, key=lambda r: r.get("published_at") or "")

    # Resolve each publication's rebalance point; drop those without prices.
    points: List[Tuple[Dict[str, Any], float]] = []
    for record in ordered:
        resolved = _rebalance_price(prices, record.get("published_at", ""))
        if resolved is not None:
            points.append((record, resolved[1]))

    result: Dict[str, Any] = {
        "evidence_class": "simulated_forward",
        "per_profile": {},
        "benchmarks": {},
        "matured_through": None,
        "cost_bps": cost_bps,
        "disclaimer": TRACK_RECORD_DISCLAIMER,
    }

    # A week matures when the *next* publication's rebalance price exists.
    matured_pairs = list(zip(points[:-1], points[1:]))
    if matured_pairs:
        result["matured_through"] = matured_pairs[-1][1][0].get("publication_id")

    for profile in PROFILES:
        weekly_returns: List[float] = []
        prev_weight = 0.0
        for (rec, p0), (_next_rec, p1) in matured_pairs:
            midpoint = ((rec.get("allocations") or {}).get(profile) or {}).get("midpoint")
            if midpoint is None or p0 <= 0:
                continue
            weight = float(midpoint) / 100.0
            gold_ret = p1 / p0 - 1.0
            turnover_cost = abs(weight - prev_weight) * cost_bps / 10000.0
            weekly_returns.append((1.0 + weight * gold_ret) * (1.0 - turnover_cost) - 1.0)
            prev_weight = weight
        result["per_profile"][profile] = {
            **_profile_stats(weekly_returns),
            "evidence_class": "simulated_forward",
        }

    # Benchmarks over the same matured span, cost-free by construction.
    if matured_pairs:
        p_first = matured_pairs[0][0][1]
        p_last = matured_pairs[-1][1][1]
        gold_cum = p_last / p_first - 1.0
        result["benchmarks"]["gold_buy_hold"] = {
            "evidence_class": "simulated_forward",
            "cum_return": round(gold_cum, 6),
            "description": "同期 100% 黄金买入持有（无成本）",
        }
        first_mid = (
            (matured_pairs[0][0][0].get("allocations") or {}).get("balanced") or {}
        ).get("midpoint")
        if first_mid is not None:
            w = float(first_mid) / 100.0
            static_returns = [
                w * (p1 / p0 - 1.0) for (_r, p0), (_n, p1) in matured_pairs
            ]
            equity = 1.0
            for r in static_returns:
                equity *= 1.0 + r
            result["benchmarks"]["static_midpoint"] = {
                "evidence_class": "simulated_forward",
                "cum_return": round(equity - 1.0, 6),
                "weight_pct": first_mid,
                "description": "首次发布的 balanced 中点恒定持有（无成本）",
            }
    else:
        result["benchmarks"]["gold_buy_hold"] = {
            "evidence_class": "simulated_forward", "cum_return": None
        }
        result["benchmarks"]["static_midpoint"] = {
            "evidence_class": "simulated_forward", "cum_return": None
        }

    return result
