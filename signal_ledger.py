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

import fcntl
import hashlib
import json
import math
import threading
from contextlib import contextmanager
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
    def append_if_absent(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]: ...


class MemoryLedgerStore:
    """In-memory store for tests and ephemeral environments."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._records.append(json.loads(json.dumps(record, ensure_ascii=False)))

    def append_if_absent(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        with self._lock:
            for existing in self._records:
                if existing["publication_id"] == record["publication_id"]:
                    return json.loads(json.dumps(existing)), False
            self._records.append(json.loads(json.dumps(record)))
            return record, True

    def load_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._records]


class JsonlLedgerStore:
    """Append-only JSONL file store; existing lines are never rewritten."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @contextmanager
    def _open_locked(self, *, writing: bool):
        with self._path.open("a+" if writing else "r", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX if writing else fcntl.LOCK_SH)
            stream.seek(0)
            yield stream
            # Closing the stream releases the lock after any buffered write.

    @staticmethod
    def _write(stream, record: Dict[str, Any]) -> None:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()

    def append(self, record: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._open_locked(writing=True) as stream:
            self._write(stream, record)

    def append_if_absent(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._open_locked(writing=True) as stream:
            for line in stream:
                if line.strip():
                    existing = json.loads(line)
                    if existing["publication_id"] == record["publication_id"]:
                        return existing, False
            self._write(stream, record)
            return record, True

    def load_all(self) -> List[Dict[str, Any]]:
        try:
            with self._open_locked(writing=False) as stream:
                return [json.loads(line) for line in stream if line.strip()]
        except FileNotFoundError:
            return []


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
        "degraded": dict(ctx.get("degraded") or {}),
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
    if ctx.get("data_stale") or (ctx.get("degraded") or {}).get("market_data"):
        for existing in store.load_all():
            if existing.get("publication_id") == publication_id_for(now):
                return existing, False
        raise ValueError("stale or unavailable research data")
    record = build_publication(ctx, now=now)
    return store.append_if_absent(record)


# --------------------------------------------------------------------------- #
# Track record scoring
# --------------------------------------------------------------------------- #
def _rebalance_price(prices: pd.Series, published_at: str) -> Optional[Tuple[pd.Timestamp, float]]:
    """First price strictly after publication. Date-only bars start at UTC midnight."""
    try:
        when = pd.Timestamp(published_at)
    except (ValueError, TypeError):
        return None
    if pd.isna(when):
        return None
    when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
    # Midnight labels carry no intraday close time. For those inputs, use a
    # later session rather than crediting the publication with same-day prices.
    pos = pd.to_datetime(prices.index, utc=True).searchsorted(when, side="right")
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


def _shadow_step(weight: float, drift: float, gold_return: float, rate: float) -> Tuple[float, float]:
    # Trade sizing and fee are both fractions of pre-trade NAV. The target
    # weight is a fraction of the smaller, post-fee NAV.
    side = 1 if weight >= drift else -1
    fee = rate * abs(weight - drift) / (1.0 + side * rate * weight)
    growth = 1.0 + weight * gold_return
    next_drift = weight * (1.0 + gold_return) / growth
    return (1.0 - fee) * growth - 1.0, next_drift


def score_track_record(
    publications: List[Dict[str, Any]],
    prices: pd.Series,
    *,
    cost_bps: float = 5.0,
    decision_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Score matured weeks of the shadow portfolios against fixed benchmarks."""
    prices = prices.astype(float).dropna().sort_index()
    if not prices.index.is_unique or not prices.map(lambda p: math.isfinite(p) and p > 0).all():
        raise ValueError("shadow prices must be positive and uniquely timestamped")
    if not 0 <= cost_bps < 10000:
        raise ValueError("invalid shadow transaction cost")
    if decision_at is not None:
        from data_loader import market_data_as_of

        prices = market_data_as_of(prices.to_frame("Gold"), decision_at)["Gold"]
    ordered = sorted(publications, key=lambda r: r.get("published_at") or "")

    # Resolve each publication's rebalance point; drop those without prices.
    executions: Dict[pd.Timestamp, Tuple[Dict[str, Any], float]] = {}
    for record in ordered:
        resolved = _rebalance_price(prices, record.get("published_at", ""))
        if resolved is not None:
            executions[resolved[0]] = (record, resolved[1])
    points = [executions[date] for date in sorted(executions)]

    result: Dict[str, Any] = {
        "evidence_class": "simulated_forward",
        "per_profile": {},
        "benchmarks": {},
        "matured_through": None,
        "cost_bps": cost_bps,
        "execution_policy": "first_close_after_publication",
        "cash_yield": 0.0,
        "turnover_method": "drift_adjusted_self_financing",
        "disclaimer": TRACK_RECORD_DISCLAIMER,
    }

    # A week matures when the *next* publication's rebalance price exists.
    matured_pairs = list(zip(points[:-1], points[1:]))
    if matured_pairs:
        result["matured_through"] = matured_pairs[-1][1][0].get("publication_id")

    for profile in PROFILES:
        weekly_returns: List[float] = []
        fixed_returns: List[float] = []
        fixed_weight = None
        drift = fixed_drift = 0.0
        for (rec, p0), (_next_rec, p1) in matured_pairs:
            midpoint = ((rec.get("allocations") or {}).get(profile) or {}).get("midpoint")
            if midpoint is None and not weekly_returns:
                continue
            weight = drift if midpoint is None else float(midpoint) / 100.0
            if not 0 <= weight <= 1:
                raise ValueError("shadow allocations must lie between 0% and 100%")
            gold_return = p1 / p0 - 1.0
            rate = cost_bps / 10000.0
            period_return, drift = _shadow_step(weight, drift, gold_return, rate)
            weekly_returns.append(period_return)
            if fixed_weight is None:
                fixed_weight = weight
            fixed_return, fixed_drift = _shadow_step(fixed_weight, fixed_drift, gold_return, rate)
            fixed_returns.append(fixed_return)
        result["per_profile"][profile] = {
            **_profile_stats(weekly_returns),
            "evidence_class": "simulated_forward",
        }
        if fixed_weight is not None:
            result["benchmarks"][f"{profile}_fixed_net"] = {
                **_profile_stats(fixed_returns),
                "evidence_class": "simulated_forward",
                "cost_bps": cost_bps,
                "weight_pct": round(fixed_weight * 100, 4),
                "description": "首期仓位固定权重，同期再平衡，计入同口径成本",
            }

    # Gross reference benchmarks over the full matured span.
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


def score_configured_track_record(
    publications: List[Dict[str, Any]], *, decision_at: datetime,
) -> Dict[str, Any]:
    from gold_sessions import load_gold_outcome_prices

    prices, source = load_gold_outcome_prices(decision_at)
    result = score_track_record(publications, prices)
    result["price_evidence"] = "session_timestamps" if source == "session_timestamps" else "date_only_proxy"
    return result
