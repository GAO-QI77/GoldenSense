"""Weekly signal publication CLI.

Freezes this ISO week's allocation-range publication into the append-only
signal ledger. Idempotent: running it twice in the same week is a no-op that
exits 0, so it is safe under cron/CI retries.

Usage:
    python3 scripts/publish_signal.py

Env:
    SIGNAL_LEDGER_DATABASE_URL or DATABASE_URL  PostgreSQL ledger outside development
    SIGNAL_LEDGER_PATH  development ledger file (default data_cache/signal_ledger.jsonl)
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_loader import load_env_file  # noqa: E402
from signal_ledger import LedgerStore, build_ledger_store, publish_weekly  # noqa: E402

load_env_file()

LOGGER = logging.getLogger("publish_signal")


def _load_ctx(decision_at: datetime) -> Dict[str, Any]:
    from research_context import shared_context

    return shared_context.get_context_at(decision_at)


def publish_once(
    *,
    store: Optional[LedgerStore] = None,
    ctx: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    send_digest_on_create: bool = False,
) -> Dict[str, Any]:
    """Publish this week's record once; returns {record, created[, digest]}."""
    decision_at = now or datetime.now(timezone.utc)
    if decision_at.tzinfo is None:
        decision_at = decision_at.replace(tzinfo=timezone.utc)
    store = store or build_ledger_store(
        app_env=os.environ.get("APP_ENV", ""),
        database_url=(
            os.environ.get("SIGNAL_LEDGER_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
        ),
        jsonl_path=os.environ.get(
            "SIGNAL_LEDGER_PATH", "data_cache/signal_ledger.jsonl"
        ),
        connect_timeout_seconds=max(
            1, int(os.environ.get("SIGNAL_LEDGER_CONNECT_TIMEOUT_SECONDS", "5"))
        ),
    )
    if ctx is None:
        ctx = _load_ctx(decision_at)
    record, created = publish_weekly(store, ctx, now=decision_at)
    result: Dict[str, Any] = {"record": record, "created": created}

    # Retention loop: a *new* publication fans out to digest subscribers.
    # Republishing the same week never re-sends. Delivery failures are
    # reported, never raised -- the ledger write is the critical path.
    if created and send_digest_on_create:
        try:
            from signal_ledger import score_configured_track_record
            from subscriptions import send_digest

            track = None
            try:
                track = score_configured_track_record(
                    store.load_all(), decision_at=decision_at,
                )
            except Exception:
                track = None
            result["digest"] = send_digest(record, track)
        except Exception as exc:
            result["digest"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def main(argv: Optional[list] = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-digest", action="store_true",
                        help="publish only; skip the subscriber digest fan-out")
    args = parser.parse_args(argv)
    try:
        result = publish_once(send_digest_on_create=not args.no_digest)
    except Exception as exc:
        LOGGER.error("signal_publish_failed error=%s:%s", type(exc).__name__, exc)
        return 1
    record = result["record"]
    LOGGER.info(
        "signal_publish %s publication_id=%s data_asof=%s degraded=%d digest=%s",
        "created" if result["created"] else "already_exists",
        record["publication_id"],
        record.get("data_asof"),
        len(record.get("degraded") or {}),
        (result.get("digest") or {}).get("transport", "skipped"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
