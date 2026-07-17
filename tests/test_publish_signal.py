"""Tests for the weekly signal publish CLI and gateway autopublish switch."""
import json
import os
from datetime import datetime, timezone

import pytest

from signal_ledger import MemoryLedgerStore
from scripts.publish_signal import publish_once, main as cli_main


def _ctx() -> dict:
    return {
        "data_source": "extended",
        "data_asof": "2026-07-13",
        "degraded": {},
        "regime_posterior": {"latest": {"calm": 0.7, "elevated": 0.25, "stress": 0.05}},
        "fair_value": {"deviation_pct": 12.0, "deviation_z": 0.9},
        "macro_factors": {"composite": 0.61, "factors_used": ["real_rate"]},
        "allocation": {
            "conservative": {"reference_range_pct": [2.0, 8.0]},
            "balanced": {"reference_range_pct": [5.0, 12.0]},
            "aggressive": {"reference_range_pct": [8.0, 18.0]},
        },
    }


MON = datetime(2026, 7, 13, 9, 0, tzinfo=timezone.utc)


def test_publish_once_creates_record():
    store = MemoryLedgerStore()
    result = publish_once(store=store, ctx=_ctx(), now=MON)
    assert result["created"] is True
    assert result["record"]["publication_id"] == "2026-W29"
    assert len(store.load_all()) == 1


def test_publish_once_idempotent_same_week():
    store = MemoryLedgerStore()
    publish_once(store=store, ctx=_ctx(), now=MON)
    second = publish_once(store=store, ctx=_ctx(), now=MON)
    assert second["created"] is False
    assert len(store.load_all()) == 1


def test_cli_main_returns_zero(tmp_path, monkeypatch):
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("SIGNAL_LEDGER_PATH", str(ledger_path))
    # Avoid the heavy research-context compute in the unit test.
    monkeypatch.setattr("scripts.publish_signal._load_ctx", _ctx)
    assert cli_main(["--no-digest"]) == 0
    lines = ledger_path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["content_hash"].startswith("sha256:")
    # Second run: idempotent, still exit 0, still one line.
    assert cli_main(["--no-digest"]) == 0
    assert len(ledger_path.read_text().strip().splitlines()) == 1
