"""Historical gold driver evidence, kept separate from trade signals."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

FLOW_SERIES = {
    "etf_net_tonnes": ("全球黄金 ETF 持仓净变化", "吨", 45),
    "cot_managed_net": ("COT Managed Money 净仓", "手", 21),
    "cot_other_net": ("COT Other Reportables 净仓", "手", 21),
    "central_bank_net_tonnes": ("央行净购金", "吨", 145),
}

WINDOW = 126
CONDITION_WINDOW = 504
MIN_CONDITION_OBS = 20


def flow_snapshot_at(path: Path | None, cutoff: datetime) -> dict:
    result = {
        series: {"status": "unavailable", "label": label, "unit": unit}
        for series, (label, unit, _) in FLOW_SERIES.items()
    }
    if path is None or not path.is_file():
        return result
    try:
        required = {"series", "observation_date", "available_at", "value"}
        rows = pd.read_csv(path, dtype=str, usecols=lambda name: name in required | {"source_url"})
        if not required <= set(rows):
            raise ValueError("required flow archive fields missing")
    except (OSError, ValueError):
        for item in result.values():
            item["status"] = "invalid_archive"
        return result
    timestamp = rows["available_at"]
    rows["available_at"] = pd.to_datetime(
        timestamp.where(timestamp.str.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}.*(?:Z|[+-]\d{2}:\d{2})", na=False)),
        utc=True, format="mixed", errors="coerce",
    )
    rows["observation_date"] = pd.to_datetime(rows["observation_date"], format="%Y-%m-%d", utc=True, errors="coerce")
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows = rows.dropna(subset=list(required))
    # The caller supplies the actual request cutoff or the historical bar cutoff.
    asof = pd.Timestamp(cutoff).tz_convert("UTC")
    rows = rows.loc[
        np.isfinite(rows["value"])
        & (rows["observation_date"] <= asof.normalize())
        & (rows["observation_date"] <= rows["available_at"])
        & (rows["available_at"] <= asof)
    ].drop_duplicates()
    keys = ["observation_date", "available_at"]
    for series, (_, _, max_age) in FLOW_SERIES.items():
        versions = rows.loc[rows["series"] == series].sort_values(keys)
        if versions.empty or versions.duplicated(keys).any():
            continue
        latest = versions.iloc[-1]
        url = latest.get("source_url")
        if isinstance(url, str):
            parts = urlsplit(url)
            if parts.scheme == "https" and parts.netloc in ("www.gold.org", "publicreporting.cftc.gov"):
                result[series]["source_url"] = url
        result[series].update(
            status="ok" if asof.normalize() - latest["observation_date"] <= pd.Timedelta(days=max_age) else "stale",
            value=float(latest["value"]),
            observation_date=str(latest["observation_date"].date()),
            available_at=latest["available_at"].isoformat(),
        )
    return result


def _oil_conditions(raw: pd.DataFrame, returns: pd.Series) -> dict | None:
    if not {"Crude_Oil", "Real_10Y"} <= set(raw):
        return None
    sample = pd.DataFrame({
        "gold": returns,
        "oil": np.sign(raw["Crude_Oil"].astype(float).diff()),
        "real_rate": np.sign(raw["Real_10Y"].astype(float).diff()),
    }).tail(CONDITION_WINDOW).replace([np.inf, -np.inf], np.nan).dropna()
    groups = []
    labels = {"up": 1, "down": -1}
    for oil, oil_sign in labels.items():
        for rate, rate_sign in labels.items():
            group = sample.loc[(sample["oil"] == oil_sign) & (sample["real_rate"] == rate_sign), "gold"]
            if len(group) >= MIN_CONDITION_OBS:
                groups.append({
                    "oil": oil, "real_rate": rate, "n": len(group),
                    "mean": float(group.mean()), "p10": float(group.quantile(0.10)),
                })
    if not groups:
        return None
    latest = sample.iloc[-1] if sample.index[-1] == raw.index[-1] else None
    return {
        "interpretation": "same_session_description",
        "oil_input": "Crude_Oil",
        "window_observations": CONDITION_WINDOW,
        "groups": groups,
        "latest": {
            key: "up" if latest[key] > 0 else "down" if latest[key] < 0 else "flat"
            for key in ("oil", "real_rate")
        } if latest is not None else None,
    }


def _rate_sensitivity(raw: pd.DataFrame, log_return: pd.Series) -> dict | None:
    if not {"Real_10Y", "USD_Index"} <= set(raw):
        return None
    rate = raw["Real_10Y"].astype(float).diff()
    usd = raw["USD_Index"].astype(float)
    sample = pd.DataFrame({
        "gold": log_return, "rate": rate, "dollar": np.log(usd.where(usd > 0)).diff(),
    }).tail(2 * WINDOW)
    if len(sample) != 2 * WINDOW or not np.isfinite(sample.to_numpy()).all():
        return None

    values = {}
    for name, part in (("prior", sample.iloc[:WINDOW]), ("recent", sample.iloc[WINDOW:])):
        design = np.column_stack((np.ones(WINDOW), part["rate"], part["dollar"]))
        coef, _, rank, _ = np.linalg.lstsq(design, part["gold"], rcond=None)
        if rank != design.shape[1]:
            return None
        values[f"{name}_log_pct_per_100bp"] = float(coef[1] * 100.0)
        values[f"{name}_span"] = [str(part.index[0].date()), str(part.index[-1].date())]
    return {
        **values,
        "window_observations": WINDOW,
        "interpretation": "same_session_sensitivity",
    }


def build_gold_drivers(raw: pd.DataFrame, *, flow_path: Path | None = None) -> dict:
    """Estimate descriptive driver relationships on one visible daily snapshot."""
    if raw.empty or not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("gold driver history must be nonempty with ordered, unique dates")
    asof = pd.to_datetime(raw.index[-1], utc=True).normalize()
    gold = raw["Gold"].astype(float)
    gold = gold.where(gold > 0)
    return {
        "data_asof": str(asof.date()),
        "oil_rate_conditions": _oil_conditions(raw, gold.pct_change(fill_method=None)),
        "rate_sensitivity": _rate_sensitivity(raw, np.log(gold).diff()),
        "flows": flow_snapshot_at(flow_path, asof + pd.Timedelta(days=1)),
    }
