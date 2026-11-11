"""Tests for the weekly signal publish CLI and gateway autopublish switch."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from signal_ledger import MemoryLedgerStore
from scripts.publish_signal import publish_once, main as cli_main


def _ctx() -> dict:
    return {
        "data_source": "extended",
        "data_asof": "2026-07-10",
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
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("SIGNAL_LEDGER_PATH", str(ledger_path))
    # Avoid the heavy research-context compute in the unit test.
    monkeypatch.setattr(
        "scripts.publish_signal._load_ctx",
        lambda cutoff: {**_ctx(), "data_asof": (cutoff.date() - timedelta(days=1)).isoformat()},
    )
    assert cli_main(["--no-digest"]) == 0
    lines = ledger_path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["content_hash"].startswith("sha256:")
    # Second run: idempotent, still exit 0, still one line.
    assert cli_main(["--no-digest"]) == 0
    assert len(ledger_path.read_text().strip().splitlines()) == 1


def test_stale_context_cannot_create_a_shadow_publication():
    store = MemoryLedgerStore()
    with pytest.raises(ValueError, match="stale"):
        publish_once(store=store, ctx={**_ctx(), "data_stale": True}, now=MON)
    assert store.load_all() == []


def test_publish_cli_uses_durable_store_factory_outside_development(monkeypatch):
    selected = MemoryLedgerStore()
    calls = []
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql://ledger.example/app")
    for key in ("SIGNAL_LEDGER_DATABASE_URL", "SIGNAL_LEDGER_PATH", "SIGNAL_LEDGER_CONNECT_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "scripts.publish_signal.build_ledger_store",
        lambda **kwargs: calls.append(kwargs) or selected,
    )

    result = publish_once(ctx=_ctx(), now=MON)

    assert result["created"] is True
    assert calls == [
        {
            "app_env": "production",
            "database_url": "postgresql://ledger.example/app",
            "jsonl_path": "data_cache/signal_ledger.jsonl",
            "connect_timeout_seconds": 5,
        }
    ]


def test_publish_uses_one_decision_cutoff_for_context_and_record(monkeypatch):
    from research_context import shared_context

    cutoffs = []
    monkeypatch.setattr(
        shared_context, "get_context_at", lambda cutoff: cutoffs.append(cutoff) or _ctx()
    )

    result = publish_once(store=MemoryLedgerStore(), now=MON)

    assert cutoffs == [MON]
    assert result["record"]["published_at"] == MON.isoformat()
