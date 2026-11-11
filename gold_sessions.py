"""Join sourced gold closes to their exchange session and visibility clocks."""
from __future__ import annotations

from datetime import datetime
import os

import numpy as np
import pandas as pd

PRICE_AVAILABILITY_ATTR = "available_at_by_close"


def _utc_timestamps(values: pd.Series) -> pd.Series:
    if not values.astype(str).str.fullmatch(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}.*(?:Z|[+-]\d{2}:\d{2})"
    ).all():
        raise ValueError("session timestamps must have an explicit UTC offset")
    return pd.to_datetime(values, utc=True, format="mixed")


def completed_gold_sessions(prices: pd.Series, calendar: pd.DataFrame, *, through: str) -> pd.DataFrame:
    """Require a one-to-one match to the supplied instrument's session calendar."""
    cutoff = _utc_timestamps(pd.Series([through])).iloc[0]
    days = pd.DatetimeIndex(pd.to_datetime(calendar["Date"], format="%Y-%m-%d"))
    close = _utc_timestamps(calendar["close_at"])
    available = _utc_timestamps(calendar["available_at"])
    if (
        not days.is_unique or not days.is_monotonic_increasing or days.hasnans
        or not close.is_monotonic_increasing or not close.is_unique
        or not available.is_monotonic_increasing or not available.is_unique
        or (available < close).any()
    ):
        raise ValueError("calendar sessions must be ordered with availability at or after close")
    expected = days[(available <= cutoff).to_numpy()]
    if expected.empty:
        raise ValueError("calendar has no completed sessions")
    observed = prices.loc[prices.index <= expected[-1]]
    if not observed.index.equals(expected):
        raise ValueError("prices do not match the completed session calendar")
    observed = observed.astype(float)
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("gold closes must be finite and positive")
    visible = available <= cutoff
    return pd.DataFrame({
        "Gold": observed.to_numpy(),
        "close_at": close.loc[visible].array,
        "available_at": available.loc[visible].array,
    }, index=expected)


def load_gold_outcome_prices(decision_at: datetime) -> tuple[pd.Series, str]:
    """Return completed prices with their availability clocks keyed by UTC close."""
    close_path = os.environ.get("GOLD_VERIFIED_CLOSES_PATH")
    session_path = os.environ.get("GOLD_VERIFIED_SESSIONS_PATH")
    if bool(close_path) != bool(session_path):
        raise ValueError("configure both verified gold price and session files")
    if close_path and session_path:
        rows = pd.read_csv(close_path, usecols=["Date", "Gold"])
        dates = pd.to_datetime(rows["Date"], format="%Y-%m-%d")
        prices = pd.Series(rows["Gold"].to_numpy(), index=pd.DatetimeIndex(dates))
        bars = completed_gold_sessions(prices, pd.read_csv(session_path), through=decision_at.isoformat())
        verified_prices = bars.set_index("close_at")["Gold"]
        verified_prices.attrs[PRICE_AVAILABILITY_ATTR] = dict(
            zip(verified_prices.index, bars["available_at"])
        )
        return verified_prices, "session_timestamps"

    from data_loader import market_availability_times, market_data_as_of
    from data_sources import load_market_data

    raw, source = load_market_data()
    dates = pd.DatetimeIndex(raw.index)
    availability_by_close = dict(
        zip(pd.to_datetime(dates, utc=True), market_availability_times(dates))
    )
    prices = market_data_as_of(raw[["Gold"]], decision_at)["Gold"]
    prices.attrs[PRICE_AVAILABILITY_ATTR] = {
        close: availability_by_close[close] for close in pd.to_datetime(prices.index, utc=True)
    }
    return prices, source
