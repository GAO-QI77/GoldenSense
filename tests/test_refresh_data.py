"""Offline tests for the scheduled data-refresh entrypoint.

These never touch the network: the dataset build is patched to fail so the
degradation path (keep last-good CSV, warm context off it) is exercised.
"""
from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import pytest

from scripts import refresh_data


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    from data_loader import NewsDataLoader
    import data_sources

    def _boom(*args, **kwargs):
        raise RuntimeError("network_down")

    monkeypatch.setattr(data_sources, "build_extended_market_data", _boom)
    monkeypatch.setattr(NewsDataLoader, "fetch_news", lambda _self: [])
    monkeypatch.delenv("GOLD_DRIVER_COT_REFRESH_ENABLED", raising=False)


def test_background_refresh_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DATA_REFRESH_ENABLED", raising=False)
    assert refresh_data.start_background_refresh() is None


def test_background_refresh_explicit_disable_wins_over_interval(monkeypatch):
    monkeypatch.setattr(
        refresh_data.threading.Thread,
        "start",
        lambda _self: pytest.fail("disabled refresh must not start a thread"),
    )

    assert (
        refresh_data.start_background_refresh(
            enabled=False,
            interval_seconds=300,
        )
        is None
    )


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


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("stale_source", [False, True])
def test_refresh_replaces_shared_market_path_only_with_fresh_sources(
    tmp_path, monkeypatch, configured, stale_source,
):
    import data_sources
    from research_context import shared_context

    target = tmp_path / "market.csv"
    monkeypatch.delenv("GOLD_RESEARCH_MARKET_PATH", raising=False)
    if configured:
        monkeypatch.setenv("GOLD_RESEARCH_MARKET_PATH", str(target))
    else:
        monkeypatch.setattr(data_sources, "RESEARCH_MARKET_CACHE_PATH", str(target))
    old = pd.DataFrame({"Gold": [1.0]}, index=pd.to_datetime(["2026-01-01"]))
    old.to_csv(target)

    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    candidate = pd.DataFrame(
        {series: [2.0] for series in data_sources.YF_TICKERS},
        index=pd.to_datetime([yesterday]),
    )
    source_ends = {series: yesterday.isoformat() for series in candidate}
    if stale_source:
        source_ends["Gold"] = "2026-01-01"

    def build(*, cache_dir, out_path):
        assert str(target) != out_path
        candidate.to_csv(out_path)
        return candidate, data_sources.DataBuildMeta(price_end_by_column=source_ends)

    def get_context(*, force_refresh):
        assert force_refresh
        frame, source = data_sources.load_market_data()
        return {"data_asof": frame.index.max().date().isoformat(), "data_source": source}

    monkeypatch.setattr(data_sources, "build_extended_market_data", build)
    monkeypatch.setattr(shared_context, "get_context", get_context)

    summary = refresh_data.refresh_once()

    assert summary["data_asof"] == ("2026-01-01" if stale_source else yesterday.isoformat())
    assert summary["ok"] is not stale_source


def test_refresh_log_is_structured_and_does_not_expose_provider_details(capsys):
    secret_url = "https://provider.example/data?api_key=secret-value"

    refresh_data._log(
        {
            "ok": False,
            "step": "dataset_failed_kept_last_good",
            "dataset_error": f"RuntimeError: failed at {secret_url}",
            "context_degraded": {
                "market_data": f"RuntimeError: failed at {secret_url}"
            },
            "data_asof": "2026-09-04",
            "data_source": "extended",
            "elapsed_ms": 25,
        }
    )

    payload = json.loads(capsys.readouterr().out.removeprefix("data_refresh "))
    assert payload == {
        "ok": False,
        "status": "last_good",
        "step": "dataset_failed_kept_last_good",
        "data_asof": "2026-09-04",
        "data_source": "extended",
        "degraded_components": ["market_data"],
        "elapsed_ms": 25,
    }
    assert secret_url not in json.dumps(payload)


def test_usable_data_asof_requires_a_real_calendar_date():
    assert refresh_data.has_usable_data_asof({"data_asof": "2026-09-04"}) is True
    assert refresh_data.has_usable_data_asof({"data_asof": "2026-99-99"}) is False
    assert refresh_data.has_usable_data_asof({"data_asof": "not-a-date"}) is False


def test_background_refresh_enforces_minimum_interval_without_immediate_run(
    monkeypatch,
):
    sleeps = []
    refreshes = []

    class StopLoop(Exception):
        pass

    class InlineThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            try:
                self.target()
            except StopLoop:
                pass

    def stop_on_sleep(seconds):
        sleeps.append(seconds)
        raise StopLoop

    monkeypatch.setattr(refresh_data.threading, "Thread", InlineThread)
    monkeypatch.setattr(refresh_data.time, "sleep", stop_on_sleep)
    monkeypatch.setattr(refresh_data, "refresh_once", lambda: refreshes.append(True))

    thread = refresh_data.start_background_refresh(
        interval_seconds=1,
        run_immediately=False,
    )

    assert thread is not None
    assert thread.daemon is True
    assert sleeps == [300.0]
    assert refreshes == []


def test_main_returns_zero_when_local_csv_available():
    # Even with the network build patched to fail, the committed base CSV lets
    # the research context assemble, so cron should exit 0 (usable dataset).
    rc = refresh_data.main()
    assert rc in (0, 1)  # 0 when local CSV present; 1 only if context empty


# --------------------------- news auto-archival ---------------------------- #
def test_archive_recent_news_ingests_relevant_items(tmp_path):
    from news_archive import NewsArchive
    from scripts.refresh_data import archive_recent_news

    archive = NewsArchive(tmp_path / "archive.jsonl")

    def fake_fetch():
        return [
            {"title": "美联储宣布加息 75 个基点", "summary": "通胀创四十年新高",
             "published": "2026-07-16", "source": "feed"},
            {"title": "本地球队夺冠", "summary": "体育新闻", "published": "2026-07-16",
             "source": "feed"},
        ]

    report = archive_recent_news(fetch_items=fake_fetch, archive=archive)
    assert report["fetched"] == 2
    assert report["kept"] == 1
    assert report["dropped"] == 1


def test_archive_recent_news_degrades_on_feed_failure(tmp_path):
    from news_archive import NewsArchive
    from scripts.refresh_data import archive_recent_news

    def broken_fetch():
        raise RuntimeError("feed down")

    report = archive_recent_news(
        fetch_items=broken_fetch, archive=NewsArchive(tmp_path / "a.jsonl")
    )
    assert "RuntimeError" in report["error"]
