"""Weekly signal publication CLI.

Freezes this ISO week's allocation-range publication into the append-only
signal ledger. Idempotent: running it twice in the same week is a no-op that
exits 0, so it is safe under cron/CI retries.

Usage:
    python3 scripts/publish_signal.py

Env:
    SIGNAL_LEDGER_PATH  ledger file (default data_cache/signal_ledger.jsonl)
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_loader import load_env_file  # noqa: E402
from signal_ledger import JsonlLedgerStore, LedgerStore, publish_weekly  # noqa: E402

load_env_file()

LOGGER = logging.getLogger("publish_signal")


def _load_ctx() -> Dict[str, Any]:
    from research_context import shared_context

    return shared_context.get_context()


def publish_once(
    *,
    store: Optional[LedgerStore] = None,
    ctx: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    send_digest_on_create: bool = False,
) -> Dict[str, Any]:
    """Publish this week's record once; returns {record, created[, digest]}."""
    store = store or JsonlLedgerStore(
        os.environ.get("SIGNAL_LEDGER_PATH", "data_cache/signal_ledger.jsonl")
    )
    ctx = ctx or _load_ctx()
    record, created = publish_weekly(store, ctx, now=now)
    result: Dict[str, Any] = {"record": record, "created": created}

    # Retention loop: a *new* publication fans out to digest subscribers.
    # Republishing the same week never re-sends. Delivery failures are
    # reported, never raised -- the ledger write is the critical path.
    if created and send_digest_on_create:
        try:
            from signal_ledger import score_track_record
            from subscriptions import send_digest

            track = None
            try:
                from data_sources import load_market_data

                raw, _source = load_market_data()
                track = score_track_record(store.load_all(), raw["Gold"])
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
