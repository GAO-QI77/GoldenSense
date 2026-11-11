"""Extended macro data layer for GoldenSense.

Adds the first-principles gold drivers that the base ``raw_market_data.csv``
lacks, while keeping the project's honesty discipline: every fetch has a local
cache, every failure degrades explicitly instead of silently.

Sources
-------
- FRED ``fredgraph.csv`` endpoint (no API key required):
    DFII10  -- 10Y TIPS real yield (the canonical macro anchor for gold)
    T10YIE  -- 10Y breakeven inflation expectation
    DGS2    -- 2Y constant-maturity nominal yield
- yfinance daily closes for the existing 8 market columns, pulled from 2004
  onwards so backtests cover more than one regime, plus GLD dollar volume as
  an *explicitly labelled* ETF flow proxy (true holdings need a paid source).

Offline behaviour
-----------------
``build_extended_market_data`` returns whatever it could assemble together
with a ``meta`` dict listing degraded sources. ``load_market_data`` falls back
to the repo's committed ``raw_market_data.csv`` when no extended file exists,
so every downstream consumer keeps working with zero network access.
"""
from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DEFAULT_CACHE_DIR = "data_cache"
EXTENDED_DATA_PATH = "raw_market_data_extended.csv"
RESEARCH_MARKET_CACHE_PATH = os.path.join(DEFAULT_CACHE_DIR, "market_extended.csv")
BASE_DATA_PATH = "raw_market_data.csv"
DEFAULT_START = "2004-01-01"

# Column name -> FRED series id.
FRED_SERIES = {
    "Real_10Y": "DFII10",
    "Breakeven_10Y": "T10YIE",
    "2Y_CMT": "DGS2",
}

# Column name -> yfinance ticker (mirrors data_loader.MarketDataLoader).
YF_TICKERS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "USD_Index": "DX-Y.NYB",
    "S&P500": "^GSPC",
    "VIX": "^VIX",
    "Crude_Oil": "CL=F",
    "10Y_Bond": "^TNX",
}


@dataclass
class DataBuildMeta:
    """Provenance record for an assembled market frame."""

    sources_ok: List[str] = field(default_factory=list)
    sources_degraded: Dict[str, str] = field(default_factory=dict)
    start: Optional[str] = None
    end: Optional[str] = None
    rows: int = 0
    price_end_by_column: Dict[str, str] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        return bool(self.sources_degraded)

    def as_dict(self) -> Dict:
        return {
            "sources_ok": list(self.sources_ok),
            "sources_degraded": dict(self.sources_degraded),
            "start": self.start,
            "end": self.end,
            "rows": self.rows,
            "degraded": self.degraded,
            "price_end_by_column": dict(self.price_end_by_column),
        }


def _cache_path(cache_dir: str, name: str) -> str:
    return os.path.join(cache_dir, f"{name}.csv")


def _read_cache(cache_dir: str, name: str, max_age_days: Optional[float]) -> Optional[pd.Series]:
    path = _cache_path(cache_dir, name)
    if not os.path.exists(path):
        return None
    if max_age_days is not None:
        age_days = (time.time() - os.path.getmtime(path)) / 86400.0
        if age_days > max_age_days:
            return None
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        return frame.iloc[:, 0].astype(float)
    except Exception:
        return None


def _write_cache(cache_dir: str, name: str, series: pd.Series) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    series.to_frame(name).to_csv(_cache_path(cache_dir, name))


def fetch_fred_series(
    series_id: str,
    *,
    start: str = DEFAULT_START,
    cache_dir: str = DEFAULT_CACHE_DIR,
    cache_name: Optional[str] = None,
    max_age_days: float = 1.0,
    timeout: float = 10.0,
) -> Optional[pd.Series]:
    """Fetch one FRED series as a float Series indexed by date.

    Order of preference: fresh cache -> network -> stale cache -> None.
    """
    name = cache_name or f"fred_{series_id}"
    cached = _read_cache(cache_dir, name, max_age_days)
    if cached is not None:
        return cached

    try:
        import requests

        resp = requests.get(
            FRED_URL,
            params={"id": series_id, "cosd": start},
            timeout=timeout,
            headers={"User-Agent": "GoldenSenseBot/1.0 (+https://localhost)"},
        )
        resp.raise_for_status()
        frame = pd.read_csv(io.StringIO(resp.text))
        date_col, value_col = frame.columns[0], frame.columns[1]
        frame[date_col] = pd.to_datetime(frame[date_col])
        series = (
            pd.to_numeric(frame.set_index(date_col)[value_col], errors="coerce")
            .dropna()
            .rename(series_id)
        )
        if series.empty:
            raise ValueError(f"FRED returned no rows for {series_id}")
        _write_cache(cache_dir, name, series)
        return series
    except Exception:
        # Stale cache beats nothing.
        return _read_cache(cache_dir, name, max_age_days=None)


def fetch_yf_closes(
    tickers: Dict[str, str],
    *,
    start: str = DEFAULT_START,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_age_days: float = 1.0,
) -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    """Fetch daily closes per ticker. Returns (series_by_column, failures)."""
    out: Dict[str, pd.Series] = {}
    failures: Dict[str, str] = {}
    for column, ticker in tickers.items():
        cached = _read_cache(cache_dir, f"yf_{column}", max_age_days)
        if cached is not None:
            out[column] = cached
            continue
        try:
            import yfinance as yf

            frame = yf.download(ticker, start=start, interval="1d", progress=False)
            if frame is None or frame.empty:
                raise ValueError("empty frame")
            closes = frame["Close"]
            if isinstance(closes, pd.DataFrame):
                closes = closes.iloc[:, 0]
            series = closes.astype(float).dropna().rename(column)
            _write_cache(cache_dir, f"yf_{column}", series)
            out[column] = series
        except Exception as exc:
            stale = _read_cache(cache_dir, f"yf_{column}", max_age_days=None)
            if stale is not None:
                out[column] = stale
            else:
                failures[column] = f"{type(exc).__name__}: {exc}"
    return out, failures


def fetch_gld_flow_proxy(
    *,
    start: str = DEFAULT_START,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_age_days: float = 1.0,
) -> Optional[pd.Series]:
    """GLD dollar volume as an ETF flow *proxy* (not true holdings data)."""
    cached = _read_cache(cache_dir, "yf_GLD_DollarVolume", max_age_days)
    if cached is not None:
        return cached
    try:
        import yfinance as yf

        frame = yf.download("GLD", start=start, interval="1d", progress=False)
        if frame is None or frame.empty:
            raise ValueError("empty frame")
        close = frame["Close"]
        volume = frame["Volume"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        proxy = (close.astype(float) * volume.astype(float)).dropna()
        proxy = proxy.rename("GLD_DollarVolume")
        _write_cache(cache_dir, "yf_GLD_DollarVolume", proxy)
        return proxy
    except Exception:
        return _read_cache(cache_dir, "yf_GLD_DollarVolume", max_age_days=None)


def merge_market_frames(
    price_series: Dict[str, pd.Series],
    macro_series: Dict[str, pd.Series],
) -> pd.DataFrame:
    """Outer-join everything on the price calendar, forward-fill macro gaps.

    Macro series (FRED) publish with a lag and skip market holidays, so they
    are forward-filled onto the trading calendar defined by the price data.
    """
    if not price_series:
        raise ValueError("at least one price series is required")
    prices = pd.DataFrame(price_series).sort_index()
    calendar = prices.index[prices["Gold"].notna()] if "Gold" in prices else prices.index
    prices = prices.ffill().loc[calendar].dropna()
    combined = prices
    for name, series in macro_series.items():
        aligned = series.sort_index().reindex(prices.index, method="ffill")
        combined = combined.join(aligned.rename(name))
    return combined


def build_extended_market_data(
    *,
    start: str = DEFAULT_START,
    cache_dir: str = DEFAULT_CACHE_DIR,
    out_path: Optional[str] = EXTENDED_DATA_PATH,
    max_age_days: float = 1.0,
) -> Tuple[pd.DataFrame, DataBuildMeta]:
    """Assemble the extended frame: long-history prices + real-rate macro."""
    meta = DataBuildMeta()

    price_series, price_failures = fetch_yf_closes(
        YF_TICKERS, start=start, cache_dir=cache_dir, max_age_days=max_age_days
    )
    meta.sources_ok.extend(sorted(price_series))
    meta.price_end_by_column = {
        name: pd.Timestamp(series.index.max()).date().isoformat()
        for name, series in price_series.items() if not series.empty
    }
    meta.sources_degraded.update(price_failures)

    macro_series: Dict[str, pd.Series] = {}
    for column, series_id in FRED_SERIES.items():
        series = fetch_fred_series(
            series_id, start=start, cache_dir=cache_dir, max_age_days=max_age_days
        )
        if series is not None:
            macro_series[column] = series
            meta.sources_ok.append(column)
        else:
            meta.sources_degraded[column] = "fred_unavailable"

    flow = fetch_gld_flow_proxy(start=start, cache_dir=cache_dir, max_age_days=max_age_days)
    if flow is not None:
        macro_series["GLD_DollarVolume"] = flow
        meta.sources_ok.append("GLD_DollarVolume")
    else:
        meta.sources_degraded["GLD_DollarVolume"] = "yfinance_unavailable"

    if "Gold" not in price_series:
        raise RuntimeError(
            "extended data build failed: no Gold price series "
            f"(failures: {price_failures})"
        )

    combined = merge_market_frames(price_series, macro_series)
    # Keep the legacy column so FeatureEngineer consumers do not break.
    if "2Y_CMT" in combined.columns and "2Y_Bond" not in combined.columns:
        combined["2Y_Bond"] = combined["2Y_CMT"]

    meta.rows = len(combined)
    meta.start = str(combined.index.min().date()) if len(combined) else None
    meta.end = str(combined.index.max().date()) if len(combined) else None

    if out_path:
        combined.to_csv(out_path, index_label="Date")
    return combined, meta


def _align_macro_vintages(frame: pd.DataFrame, path: str) -> pd.DataFrame:
    # The legacy extended file aliases FRED DGS2 as 2Y_Bond.
    frame = frame.drop(columns=[*FRED_SERIES, "2Y_Bond"], errors="ignore")
    try:
        vintages = pd.read_csv(path, dtype=str, usecols=[
            "series_id", "observation_date", "available_at", "value",
        ])
    except (OSError, ValueError) as exc:
        frame.attrs["macro_vintages_error"] = type(exc).__name__
        return frame

    # Date-only releases are ambiguous. Require a clock and explicit UTC offset.
    offset = r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}.*(?:Z|[+-]\d{2}:\d{2})$"
    vintages["available_at"] = pd.to_datetime(
        vintages["available_at"].where(vintages["available_at"].str.match(offset, na=False)),
        utc=True, format="mixed", errors="coerce",
    )
    observation_date = vintages["observation_date"]
    vintages["observation_date"] = pd.to_datetime(
        observation_date.where(observation_date.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)),
        utc=True, format="%Y-%m-%d", errors="coerce",
    )
    vintages["value"] = pd.to_numeric(vintages["value"], errors="coerce")
    vintages = vintages.dropna()
    vintages = vintages.loc[
        np.isfinite(vintages["value"])
        & (vintages["observation_date"] <= vintages["available_at"])
    ]
    from data_loader import market_availability_times

    decisions = market_availability_times(pd.DatetimeIndex(frame.index))
    keys = ["available_at", "observation_date"]
    for name, series_id in FRED_SERIES.items():
        series = vintages.loc[vintages["series_id"] == series_id].drop_duplicates()
        if series.empty or series.duplicated(keys).any():
            continue
        series = series.sort_values(keys)
        # A revision changes the active reading only if it revises the latest
        # observation available at that time. A late older release cannot replace it.
        series = series.loc[series["observation_date"].eq(series["observation_date"].cummax())]
        timeline = series.drop_duplicates("available_at", keep="last").set_index("available_at")
        aligned = timeline["value"].reindex(decisions, method="ffill")
        if aligned.notna().any():
            frame[name] = aligned.to_numpy()
    if "2Y_CMT" in frame:
        frame["2Y_Bond"] = frame["2Y_CMT"]
    return frame


def load_market_data(
    *,
    prefer_extended: bool = True,
    extended_path: Optional[str] = None,
    base_path: str = BASE_DATA_PATH,
    require_macro_vintages: bool = False,
    macro_vintage_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Load the best locally available market frame.

    Returns (frame, source) where source is ``extended`` or ``base``. Never
    touches the network -- run ``python3 data_sources.py`` to refresh.
    """
    if prefer_extended and extended_path is None:
        configured = os.environ.get("GOLD_RESEARCH_MARKET_PATH")
        if configured and not os.path.isfile(configured):
            raise FileNotFoundError("configured research market file is unavailable")
        extended_path = configured or (
            RESEARCH_MARKET_CACHE_PATH if os.path.isfile(RESEARCH_MARKET_CACHE_PATH) else EXTENDED_DATA_PATH
        )
    if prefer_extended and extended_path and os.path.isfile(extended_path):
        frame = pd.read_csv(extended_path, index_col=0, parse_dates=True).sort_index()
        frame, source = frame.dropna(subset=["Gold"]).ffill(), "extended"
    else:
        frame = pd.read_csv(base_path, index_col=0, parse_dates=True).sort_index()
        frame, source = frame.dropna(subset=["Gold"]).ffill().dropna(), "base"
    if require_macro_vintages:
        vintage_path = macro_vintage_path or os.environ.get(
            "MACRO_VINTAGES_PATH", os.path.join(DEFAULT_CACHE_DIR, "macro_vintages.csv")
        )
        frame = _align_macro_vintages(frame, vintage_path)
    return frame, source


if __name__ == "__main__":
    frame, meta = build_extended_market_data()
    print(f"rows={meta.rows} span={meta.start} -> {meta.end}")
    print(f"ok: {meta.sources_ok}")
    if meta.degraded:
        print(f"degraded: {meta.sources_degraded}")
    print(frame.tail(3).round(3).to_string())
