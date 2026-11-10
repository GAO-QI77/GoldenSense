"""Scheduled data refresh for GoldenSense.

Rebuilds the extended market dataset (FRED real rates + long-history prices)
and warms the local quant research context so `/quant` and the analyst
committee never drift far from the live market.

Two ways to run:
  1. As a cron job (Railway cron / GitHub Actions):  python3 scripts/refresh_data.py
  2. Embedded: ``start_background_refresh()`` spawns a daemon thread that
     refreshes on an interval -- used by the single-container public stack so
     the demo self-refreshes without a separate cron service.

Honesty discipline: a failed fetch degrades explicitly (keeps the last good
CSV, logs the reason) instead of crashing the deploy or fabricating data.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys
import threading
from tempfile import NamedTemporaryFile
import time
from typing import Optional

# Allow running as `python3 scripts/refresh_data.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_SAFE_DATA_ASOF = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SAFE_STEPS = {
    "start",
    "dataset_built",
    "dataset_failed_kept_last_good",
    "preflight_exception",
}


def has_usable_data_asof(summary: dict) -> bool:
    """Return whether a refresh summary identifies usable local context."""
    value = summary.get("data_asof")
    if not isinstance(value, str) or _SAFE_DATA_ASOF.fullmatch(value) is None:
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def safe_refresh_summary(summary: dict) -> dict:
    """Reduce an internal refresh result to fields safe for public logs."""
    usable = has_usable_data_asof(summary)
    step = summary.get("step")
    if step not in _SAFE_STEPS:
        step = "unknown"
    if summary.get("ok") and usable:
        status = "fresh"
    elif step == "dataset_failed_kept_last_good" and usable:
        status = "last_good"
    elif usable:
        status = "degraded"
    else:
        status = "unavailable"

    safe = {
        "ok": bool(summary.get("ok")),
        "status": status,
        "step": step,
    }
    if usable:
        safe["data_asof"] = summary["data_asof"]
    data_source = summary.get("data_source")
    if isinstance(data_source, str) and _SAFE_NAME.fullmatch(data_source):
        safe["data_source"] = data_source
    degraded = summary.get("context_degraded")
    if isinstance(degraded, dict):
        components = sorted(
            key
            for key in degraded
            if isinstance(key, str) and _SAFE_NAME.fullmatch(key)
        )
        if components:
            safe["degraded_components"] = components
    elapsed_ms = summary.get("elapsed_ms")
    if isinstance(elapsed_ms, int) and elapsed_ms >= 0:
        safe["elapsed_ms"] = elapsed_ms
    return safe


def refresh_once(*, cache_dir: Optional[str] = None) -> dict:
    """Rebuild the extended dataset and warm the research context once.

    Returns a JSON-able summary; never raises for expected network failures.
    """
    started = time.time()
    summary: dict = {"ok": False, "step": "start"}
    dataset_built = False
    try:
        from data_sources import DEFAULT_CACHE_DIR, RESEARCH_MARKET_CACHE_PATH, YF_TICKERS, build_extended_market_data

        target = Path(os.environ.get("GOLD_RESEARCH_MARKET_PATH") or RESEARCH_MARKET_CACHE_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, suffix=".csv", delete=False) as candidate:
            path = Path(candidate.name)
        try:
            frame, meta = build_extended_market_data(cache_dir=cache_dir or DEFAULT_CACHE_DIR, out_path=str(path))
            required = set(YF_TICKERS)
            if not required <= set(frame) or any(key in meta.sources_degraded for key in required):
                raise ValueError("required market sources missing")
            fresh_after = (datetime.now(timezone.utc).date() - timedelta(days=4)).isoformat()
            stale = sorted(key for key in required if meta.price_end_by_column.get(key, "") < fresh_after)
            if frame.empty or stale:
                raise ValueError(f"market refresh is stale: {', '.join(stale)}")
            path.replace(target)
        finally:
            path.unlink(missing_ok=True)
        summary["dataset"] = meta.as_dict()
        summary["step"] = "dataset_built"
        dataset_built = True
    except Exception as exc:  # network / provider failure -> keep last good CSV
        summary["dataset_error"] = f"{type(exc).__name__}: {exc}"
        summary["step"] = "dataset_failed_kept_last_good"

    if os.environ.get("GOLD_DRIVER_COT_REFRESH_ENABLED") == "1":
        try:
            from scripts.refresh_gold_driver_flows import cot_observations, fetch_cot, merge_observations

            payload = fetch_cot()
            path = Path(os.environ.get("GOLD_DRIVER_FLOWS_PATH") or "data_cache/gold_driver_flows.csv")
            summary["cot_versions_added"] = merge_observations(
                path, cot_observations(payload), captured_at=datetime.fromisoformat(payload["captured_at"]),
            )
        except Exception as exc:
            summary["cot_refresh_error"] = type(exc).__name__

    # Warm (force-refresh) the shared research context off whatever CSV we now
    # have, so the first user request after a refresh is already hot.
    try:
        from research_context import shared_context

        ctx = shared_context.get_context(force_refresh=True)
        summary["context_degraded"] = ctx.get("degraded", {})
        summary["data_asof"] = ctx.get("data_asof")
        summary["data_source"] = ctx.get("data_source")
        summary["ok"] = dataset_built and "market_data" not in ctx.get(
            "degraded", {}
        )
    except Exception as exc:
        summary["context_error"] = f"{type(exc).__name__}: {exc}"

    # Decision-relevant news accrues into the archive on every refresh, so
    # the RAG corpus builds itself over time. Never blocks the main flow.
    summary["news_archive"] = archive_recent_news()

    summary["elapsed_ms"] = int((time.time() - started) * 1000)
    return summary


def archive_recent_news(*, fetch_items=None, archive=None) -> dict:
    """Fetch recent RSS items and ingest them into the news archive.

    ``fetch_items``/``archive`` are injectable for tests. Failures degrade to
    a summary dict -- the dataset refresh must never fail because a feed is
    down.
    """
    try:
        if archive is None:
            from news_archive import NewsArchive

            archive = NewsArchive(
                os.environ.get("NEWS_ARCHIVE_PATH", "data_cache/news_archive.jsonl")
            )
        if fetch_items is None:
            from data_loader import NewsDataLoader

            fetch_items = NewsDataLoader().fetch_news
        items = fetch_items() or []
        normalized = [
            {
                "title": item.get("title"),
                "summary": item.get("summary"),
                "source": item.get("source"),
                "published_at": item.get("published_at") or item.get("published"),
            }
            for item in items
        ]
        report = archive.ingest(normalized)
        report["fetched"] = len(normalized)
        return report
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _log(summary: dict) -> None:
    print(
        "data_refresh "
        + json.dumps(safe_refresh_summary(summary), ensure_ascii=False),
        flush=True,
    )


def start_background_refresh(
    *,
    interval_seconds: Optional[float] = None,
    run_immediately: bool = False,
    enabled: Optional[bool] = None,
) -> Optional[threading.Thread]:
    """Spawn a daemon thread that refreshes on an interval. Returns the thread.

    Controlled by env so the public stack can opt in:
      DATA_REFRESH_ENABLED=1                 -> enable the embedded scheduler
      DATA_REFRESH_INTERVAL_SECONDS=86400    -> cadence (default daily)
    Returns None when disabled.
    """
    is_enabled = (
        os.environ.get("DATA_REFRESH_ENABLED", "0") == "1"
        if enabled is None
        else enabled
    )
    if enabled is False:
        return None
    if not is_enabled and interval_seconds is None:
        return None
    interval = float(
        interval_seconds
        if interval_seconds is not None
        else os.environ.get("DATA_REFRESH_INTERVAL_SECONDS", "86400")
    )
    interval = max(300.0, interval)  # never hammer the free providers

    def _loop() -> None:
        if run_immediately:
            _log(refresh_once())
        while True:
            time.sleep(interval)
            try:
                _log(refresh_once())
            except Exception as exc:  # defensive: keep the thread alive
                _log({"ok": False, "loop_error": f"{type(exc).__name__}: {exc}"})

    thread = threading.Thread(target=_loop, name="data-refresh", daemon=True)
    thread.start()
    return thread


def main() -> int:
    summary = refresh_once()
    _log(summary)
    # Cron exit code: 0 if we have a usable dataset (even if degraded to cache),
    # 1 only when the context could not be assembled at all.
    return 0 if summary.get("ok") or summary.get("data_asof") else 1


if __name__ == "__main__":
    raise SystemExit(main())
