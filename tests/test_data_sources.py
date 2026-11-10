"""Offline tests for the extended data layer (no network access needed)."""
import numpy as np
import pandas as pd
import pytest

from data_sources import (
    DataBuildMeta,
    load_market_data,
    merge_market_frames,
)


def _series(start: str, n: int, value: float = 100.0, freq: str = "B") -> pd.Series:
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.Series(np.linspace(value, value * 1.1, n), index=idx)


def test_merge_market_frames_aligns_macro_to_trading_calendar():
    prices = {"Gold": _series("2024-01-01", 30), "USD_Index": _series("2024-01-01", 30, 104.0)}
    # Macro series is sparse (weekly) and lagged -- must be ffilled onto price days.
    macro_idx = pd.date_range("2024-01-01", periods=5, freq="W-FRI")
    macro = {"Real_10Y": pd.Series([1.8, 1.9, 2.0, 2.1, 2.2], index=macro_idx)}

    combined = merge_market_frames(prices, macro)

    assert list(combined.columns) == ["Gold", "USD_Index", "Real_10Y"]
    assert len(combined) == 30
    # After the first macro print arrives, there must be no NaN gaps.
    first_valid = combined["Real_10Y"].first_valid_index()
    assert combined.loc[first_valid:, "Real_10Y"].notna().all()


def test_merge_market_frames_requires_prices():
    with pytest.raises(ValueError):
        merge_market_frames({}, {})


def test_data_build_meta_degradation_flag():
    meta = DataBuildMeta(sources_ok=["Gold"], sources_degraded={"Real_10Y": "fred_unavailable"})
    assert meta.degraded is True
    payload = meta.as_dict()
    assert payload["sources_degraded"]["Real_10Y"] == "fred_unavailable"


def test_load_market_data_falls_back_to_base_csv(tmp_path):
    base = tmp_path / "base.csv"
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    pd.DataFrame({"Gold": np.linspace(2000, 2100, 10)}, index=idx).to_csv(
        base, index_label="Date"
    )

    frame, source = load_market_data(
        prefer_extended=True,
        extended_path=str(tmp_path / "missing_extended.csv"),
        base_path=str(base),
    )
    assert source == "base"
    assert "Gold" in frame.columns and len(frame) == 10


def test_load_market_data_prefers_extended_when_present(tmp_path):
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    extended = tmp_path / "ext.csv"
    pd.DataFrame(
        {"Gold": np.linspace(2000, 2100, 10), "Real_10Y": np.linspace(1.5, 2.0, 10)},
        index=idx,
    ).to_csv(extended, index_label="Date")

    frame, source = load_market_data(
        prefer_extended=True,
        extended_path=str(extended),
        base_path=str(tmp_path / "unused.csv"),
    )
    assert source == "extended"
    assert "Real_10Y" in frame.columns


def test_research_market_data_uses_only_released_macro_vintages(tmp_path):
    dates = pd.date_range("2024-01-01", periods=5)
    extended = tmp_path / "extended.csv"
    pd.DataFrame(
        {"Gold": [100, 101, 102, 103, 104], "Real_10Y": [99] * 5, "2Y_Bond": [99] * 5},
        index=dates,
    ).to_csv(extended)
    vintages = tmp_path / "macro.csv"
    vintages.write_text(
        "series_id,observation_date,available_at,value\n"
        "DFII10,2024-01-01,2024-01-02T15:00:00Z,1\n"
        "DFII10,2024-01-01,2024-01-03T15:00:00Z,2\n"
        "DFII10,2024-01-02,2024-01-04T00:00:00-05:00,3\n"
        "DFII10,2024-01-01,2024-01-05T15:00:00Z,4\n"
        "T10YIE,2024-01-01,2024-01-02,8\n"
    )

    frame, _ = load_market_data(
        extended_path=str(extended), require_macro_vintages=True, macro_vintage_path=vintages
    )

    assert "2Y_Bond" not in frame
    assert "Breakeven_10Y" not in frame
    assert pd.isna(frame["Real_10Y"].iloc[0])
    assert frame["Real_10Y"].iloc[1:].tolist() == [1, 2, 3, 3]

    frame, _ = load_market_data(
        extended_path=str(extended), require_macro_vintages=True,
        macro_vintage_path=tmp_path / "missing.csv",
    )
    assert list(frame.columns) == ["Gold"]


def test_merge_market_frames_never_creates_a_gold_close_on_another_market_day():
    dates = pd.date_range("2024-01-01", periods=3)
    combined = merge_market_frames(
        {
            "Gold": pd.Series([100.0, 101.0], index=dates[[0, 2]]),
            "USD_Index": pd.Series([103.0, 104.0], index=dates[[0, 1]]),
        },
        {},
    )
    assert combined.index.tolist() == dates[[0, 2]].tolist()
    assert combined["USD_Index"].tolist() == [103.0, 104.0]
