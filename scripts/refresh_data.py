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

import json
import os
import sys
import threading
import time
from typing import Optional

# Allow running as `python3 scripts/refresh_data.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def refresh_once(*, cache_dir: Optional[str] = None) -> dict:
    """Rebuild the extended dataset and warm the research context once.

    Returns a JSON-able summary; never raises for expected network failures.
    """
    started = time.time()
    summary: dict = {"ok": False, "step": "start"}
    try:
        from data_sources import DEFAULT_CACHE_DIR, build_extended_market_data

        _frame, meta = build_extended_market_data(
            cache_dir=cache_dir or DEFAULT_CACHE_DIR
        )
        summary["dataset"] = meta.as_dict()
        summary["step"] = "dataset_built"
    except Exception as exc:  # network / provider failure -> keep last good CSV
        summary["dataset_error"] = f"{type(exc).__name__}: {exc}"
        summary["step"] = "dataset_failed_kept_last_good"

    # Warm (force-refresh) the shared research context off whatever CSV we now
    # have, so the first user request after a refresh is already hot.
    try:
        from research_context import shared_context

        ctx = shared_context.get_context(force_refresh=True)
        summary["context_degraded"] = ctx.get("degraded", {})
        summary["data_asof"] = ctx.get("data_asof")
        summary["data_source"] = ctx.get("data_source")
        summary["ok"] = "market_data" not in ctx.get("degraded", {})
    except Exception as exc:
        summary["context_error"] = f"{type(exc).__name__}: {exc}"

    summary["elapsed_ms"] = int((time.time() - started) * 1000)
    return summary


def _log(summary: dict) -> None:
    print(
        "data_refresh " + json.dumps(summary, ensure_ascii=False, default=str),
        flush=True,
    )


def start_background_refresh(
    *,
    interval_seconds: Optional[float] = None,
    run_immediately: bool = False,
) -> Optional[threading.Thread]:
    """Spawn a daemon thread that refreshes on an interval. Returns the thread.

    Controlled by env so the public stack can opt in:
      DATA_REFRESH_ENABLED=1                 -> enable the embedded scheduler
      DATA_REFRESH_INTERVAL_SECONDS=86400    -> cadence (default daily)
    Returns None when disabled.
    """
    enabled = os.environ.get("DATA_REFRESH_ENABLED", "0") == "1"
    if not enabled and interval_seconds is None:
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
