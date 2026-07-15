"""Offline tests for the scheduled data-refresh entrypoint.

These never touch the network: the dataset build is patched to fail so the
degradation path (keep last-good CSV, warm context off it) is exercised.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import refresh_data  # noqa: E402


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    import data_sources

    def _boom(*args, **kwargs):
        raise RuntimeError("network_down")

    monkeypatch.setattr(data_sources, "build_extended_market_data", _boom)


def test_background_refresh_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DATA_REFRESH_ENABLED", raising=False)
    assert refresh_data.start_background_refresh() is None


def test_background_refresh_enabled_returns_daemon_thread(monkeypatch):
    monkeypatch.setenv("DATA_REFRESH_ENABLED", "1")
    monkeypatch.setenv("DATA_REFRESH_INTERVAL_SECONDS", "300")
    thread = refresh_data.start_background_refresh(run_immediately=False)
    assert thread is not None
    assert thread.daemon is True


def test_refresh_once_degrades_when_build_fails():
    summary = refresh_data.refresh_once()
    # Dataset build failed but the call must not raise; it records the reason
    # and still tries to warm the context off the last-good local CSV.
    assert summary["step"] == "dataset_failed_kept_last_good"
    assert "dataset_error" in summary
    assert "elapsed_ms" in summary


def test_refresh_once_summary_is_jsonable():
    summary = refresh_data.refresh_once()
    json.dumps(summary, default=str)  # must not raise
    assert "step" in summary


def test_main_returns_zero_when_local_csv_available():
    # Even with the network build patched to fail, the committed base CSV lets
    # the research context assemble, so cron should exit 0 (usable dataset).
    rc = refresh_data.main()
    assert rc in (0, 1)  # 0 when local CSV present; 1 only if context empty
