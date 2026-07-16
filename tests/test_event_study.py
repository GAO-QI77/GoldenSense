"""Tests for the curated event catalog and the event-study library."""
from datetime import date

import pandas as pd
import pytest

from event_study import (
    CATEGORIES,
    EventStudyLibrary,
    compute_event_study,
    load_catalog,
)


# --------------------------------------------------------------------------- #
# Catalog integrity
# --------------------------------------------------------------------------- #
def test_real_catalog_loads_and_validates():
    catalog = load_catalog()
    assert len(catalog) >= 40
    ids = [e["event_id"] for e in catalog]
    assert len(ids) == len(set(ids)), "event_id must be unique"
    for event in catalog:
        assert event["category"] in CATEGORIES, event["event_id"]
        # Date parses and is within the dataset era (2004+).
        d = date.fromisoformat(event["date"])
        assert d.year >= 2004, event["event_id"]
        assert event["title"] and event["summary"]


def test_catalog_covers_all_categories():
    catalog = load_catalog()
    covered = {e["category"] for e in catalog}
    assert covered == set(CATEGORIES)


# --------------------------------------------------------------------------- #
# Forward-return computation (synthetic, hand-checked)
# --------------------------------------------------------------------------- #
def _prices() -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=260)
    # +0.1% per business day, deterministic.
    return pd.Series([1000.0 * (1.001 ** i) for i in range(len(idx))], index=idx)


def _mini_catalog() -> list:
    return [
        {"event_id": "e1", "date": "2020-02-03", "category": "geopolitics",
         "title": "t", "summary": "s"},
        {"event_id": "e2", "date": "2020-03-02", "category": "geopolitics",
         "title": "t", "summary": "s"},
        {"event_id": "e3", "date": "2019-06-01", "category": "usd",
         "title": "before data", "summary": "s"},
        {"event_id": "e4", "date": "2020-12-15", "category": "usd",
         "title": "no 90d ahead", "summary": "s"},
    ]


def test_forward_returns_hand_checked():
    study = compute_event_study(_mini_catalog(), _prices())
    e1 = next(e for e in study["events"] if e["event_id"] == "e1")
    # Deterministic +0.1%/bd: 5 business days -> 1.001^5 - 1.
    assert e1["fwd_5d"] == pytest.approx(1.001 ** 5 - 1, rel=1e-6)
    assert e1["fwd_30d"] == pytest.approx(1.001 ** 30 - 1, rel=1e-6)
    assert e1["fwd_90d"] == pytest.approx(1.001 ** 90 - 1, rel=1e-6)


def test_events_outside_data_are_none_and_excluded_from_aggregates():
    study = compute_event_study(_mini_catalog(), _prices())
    e3 = next(e for e in study["events"] if e["event_id"] == "e3")
    assert e3["fwd_30d"] is None  # anchor before data start -> not scored
    e4 = next(e for e in study["events"] if e["event_id"] == "e4")
    assert e4["fwd_90d"] is None  # not enough forward history

    geo = study["by_category"]["geopolitics"]
    assert geo["n_30d"] == 2
    assert geo["mean_30d"] == pytest.approx(1.001 ** 30 - 1, abs=1e-6)
    assert geo["positive_share_30d"] == 1.0


def test_library_analogs_with_citations():
    lib = EventStudyLibrary(catalog=_mini_catalog(), prices=_prices())
    analogs = lib.analogs_for("geopolitics")
    assert analogs["n_30d"] == 2
    assert analogs["mean_30d"] > 0
    refs = analogs["recent_events"]
    assert refs and all({"event_id", "date", "title"} <= set(r) for r in refs)
    # Unknown category degrades to None, never invents.
    assert lib.analogs_for("nonexistent") is None
