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

import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DEFAULT_CACHE_DIR = "data_cache"
EXTENDED_DATA_PATH = "raw_market_data_extended.csv"
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
    prices = prices.ffill().dropna()
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


def load_market_data(
    *,
    prefer_extended: bool = True,
    extended_path: str = EXTENDED_DATA_PATH,
    base_path: str = BASE_DATA_PATH,
) -> Tuple[pd.DataFrame, str]:
    """Load the best locally available market frame.

    Returns (frame, source) where source is ``extended`` or ``base``. Never
    touches the network -- run ``python3 data_sources.py`` to refresh.
    """
    if prefer_extended and os.path.exists(extended_path):
        frame = pd.read_csv(extended_path, index_col=0, parse_dates=True)
        return frame.ffill().dropna(subset=["Gold"]), "extended"
    frame = pd.read_csv(base_path, index_col=0, parse_dates=True)
    return frame.ffill().dropna(), "base"


if __name__ == "__main__":
    frame, meta = build_extended_market_data()
    print(f"rows={meta.rows} span={meta.start} -> {meta.end}")
    print(f"ok: {meta.sources_ok}")
    if meta.degraded:
        print(f"degraded: {meta.sources_degraded}")
    print(frame.tail(3).round(3).to_string())
