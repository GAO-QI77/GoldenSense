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
) -> Dict[str, Any]:
    """Publish this week's record once; returns {record, created}."""
    store = store or JsonlLedgerStore(
        os.environ.get("SIGNAL_LEDGER_PATH", "data_cache/signal_ledger.jsonl")
    )
    ctx = ctx or _load_ctx()
    record, created = publish_weekly(store, ctx, now=now)
    return {"record": record, "created": created}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        result = publish_once()
    except Exception as exc:
        LOGGER.error("signal_publish_failed error=%s:%s", type(exc).__name__, exc)
        return 1
    record = result["record"]
    LOGGER.info(
        "signal_publish %s publication_id=%s data_asof=%s degraded=%d",
        "created" if result["created"] else "already_exists",
        record["publication_id"],
        record.get("data_asof"),
        len(record.get("degraded") or {}),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
