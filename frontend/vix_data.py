from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass(frozen=True)
class VixPoint:
    timestamp: datetime
    value: float


@dataclass(frozen=True)
class VixSnapshot:
    timestamp: datetime
    value: float
    source: str


class VixDataError(RuntimeError):
    pass


def fetch_vix_latest(timeout_s: float = 6.0) -> VixSnapshot:
    try:
        return _fetch_vix_latest_yahoo(timeout_s=timeout_s)
    except Exception as e:
        raise VixDataError(f"failed_to_fetch_vix_latest: {type(e).__name__}: {e}") from e


def fetch_vix_history(range_: str = "6mo", interval: str = "1d", timeout_s: float = 8.0) -> Tuple[str, List[VixPoint]]:
    try:
        return _fetch_vix_history_yahoo(range_=range_, interval=interval, timeout_s=timeout_s)
    except Exception as e:
        raise VixDataError(f"failed_to_fetch_vix_history: {type(e).__name__}: {e}") from e


def validate_vix_value(value: float) -> float:
    if not (0.0 < float(value) < 200.0):
        raise VixDataError(f"vix_out_of_range: {value}")
    return float(value)


def _fetch_vix_latest_yahoo(timeout_s: float) -> VixSnapshot:
    symbol = "%5EVIX"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1m"
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    payload: Dict[str, Any] = resp.json()
    result = payload.get("chart", {}).get("result")
    if not isinstance(result, list) or not result:
        raise VixDataError("yahoo_chart_missing_result")

    r0 = result[0]
    ts_list = r0.get("timestamp")
    close_list = (
        r0.get("indicators", {})
        .get("quote", [{}])[0]
        .get("close")
    )
    if not isinstance(ts_list, list) or not isinstance(close_list, list):
        raise VixDataError("yahoo_chart_missing_series")

    for ts, close in zip(reversed(ts_list), reversed(close_list)):
        if close is None:
            continue
        v = validate_vix_value(float(close))
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        return VixSnapshot(timestamp=dt, value=v, source="YahooFinance")

    raise VixDataError("yahoo_chart_no_valid_close")


def _fetch_vix_history_yahoo(range_: str, interval: str, timeout_s: float) -> Tuple[str, List[VixPoint]]:
    symbol = "%5EVIX"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval={interval}"
    resp = requests.get(url, timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    payload: Dict[str, Any] = resp.json()
    result = payload.get("chart", {}).get("result")
    if not isinstance(result, list) or not result:
        raise VixDataError("yahoo_chart_missing_result")

    r0 = result[0]
    ts_list = r0.get("timestamp")
    close_list = (
        r0.get("indicators", {})
        .get("quote", [{}])[0]
        .get("close")
    )
    if not isinstance(ts_list, list) or not isinstance(close_list, list):
        raise VixDataError("yahoo_chart_missing_series")

    out: List[VixPoint] = []
    for ts, close in zip(ts_list, close_list):
        if close is None:
            continue
        v = validate_vix_value(float(close))
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        out.append(VixPoint(timestamp=dt, value=v))

    if not out:
        raise VixDataError("yahoo_chart_empty_history")
    return "YahooFinance", out

