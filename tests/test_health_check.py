"""Tests for the cron-ready operational health check."""
from datetime import datetime, timezone

from scripts.health_check import run_checks

# Tuesday: past the Monday-noon publication grace window.
TUESDAY = datetime(2026, 7, 14, 9, 0, tzinfo=timezone.utc)  # ISO 2026-W29
MONDAY_EARLY = datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc)


def _fetch_factory(overrides=None, unreachable=()):
    responses = {
        "/health": {"status": "ok"},
        "/api/v1/agent/research/current": {"data_age_days": 1},
        "/metrics": {"status_classes": {"2xx": 95, "4xx": 4, "5xx": 1}},
        "/api/v1/signals/current": {"publication_id": "2026-W29"},
    }
    responses.update(overrides or {})

    def fetch(url):
        for suffix, payload in responses.items():
            if url.endswith(suffix):
                if suffix in unreachable:
                    raise ConnectionError("down")
                return payload
        raise AssertionError(f"unexpected url {url}")

    return fetch


def test_all_green():
    report = run_checks(fetch=_fetch_factory(), now=TUESDAY)
    assert report["ok"] is True
    assert report["failures"] == []
    assert report["checks"]["error_rate_5xx"] == 0.01


def test_stale_data_fails():
    fetch = _fetch_factory({"/api/v1/agent/research/current": {"data_age_days": 9}})
    report = run_checks(fetch=fetch, now=TUESDAY)
    assert report["ok"] is False
    assert any("data_stale" in f for f in report["failures"])


def test_high_error_rate_fails():
    fetch = _fetch_factory({"/metrics": {"status_classes": {"2xx": 80, "5xx": 20}}})
    report = run_checks(fetch=fetch, now=TUESDAY)
    assert any("error_rate_5xx" in f for f in report["failures"])


def test_missing_publication_fails_after_grace_only():
    fetch = _fetch_factory({"/api/v1/signals/current": {"publication_id": "2026-W28"}})
    late = run_checks(fetch=fetch, now=TUESDAY)
    assert any("publication_missing" in f for f in late["failures"])
    # Monday before noon UTC: still inside the publish window -> no alarm.
    early = run_checks(fetch=fetch, now=MONDAY_EARLY)
    assert not any("publication_missing" in f for f in early["failures"])


def test_gateway_unreachable_fails():
    report = run_checks(fetch=_fetch_factory(unreachable=("/health",)), now=TUESDAY)
    assert report["checks"]["gateway"] == "unreachable"
    assert report["ok"] is False
