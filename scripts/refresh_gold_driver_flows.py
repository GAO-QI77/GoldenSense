"""Capture COMEX gold positioning and reviewed WGC excerpts without backdating."""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gold_drivers import FLOW_SERIES

COT_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
FIELDS = ("series", "observation_date", "available_at", "value", "source_url")


def fetch_cot() -> dict:
    response = requests.get(COT_URL, params={
        "cftc_contract_market_code": "088691",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": 3,
    }, timeout=20)
    response.raise_for_status()
    return {"captured_at": datetime.now(timezone.utc).isoformat(), "rows": response.json()}


def cot_observations(payload: dict) -> list[dict]:
    output = []
    dates = set()
    for row in payload["rows"]:
        if row["cftc_contract_market_code"] != "088691":
            raise ValueError("unexpected COT contract")
        day = date.fromisoformat(row["report_date_as_yyyy_mm_dd"][:10]).isoformat()
        if day in dates:
            raise ValueError("ambiguous COT report period")
        dates.add(day)
        for series, long, short in (
            ("cot_managed_net", "m_money_positions_long_all", "m_money_positions_short_all"),
            ("cot_other_net", "other_rept_positions_long", "other_rept_positions_short"),
        ):
            output.append({
                "series": series, "observation_date": day,
                "value": int(row[long]) - int(row[short]), "source_url": COT_URL,
            })
    if not output:
        raise ValueError("empty COMEX gold COT result")
    return output


def merge_observations(path: Path, records: list[dict], *, captured_at: datetime) -> int:
    if captured_at.tzinfo is None or captured_at > datetime.now(timezone.utc):
        raise ValueError("capture time must be timezone-aware and already observed")
    stamp = captured_at.astimezone(timezone.utc)
    clean = {}
    for item in records:
        series = item["series"]
        day = date.fromisoformat(item["observation_date"])
        value = float(item["value"])
        if series not in FLOW_SERIES or day > stamp.date() or not math.isfinite(value):
            raise ValueError("invalid driver observation")
        url = item.get("source_url", "")
        parsed = urlsplit(url)
        if url and (parsed.scheme != "https" or parsed.netloc not in ("www.gold.org", "publicreporting.cftc.gov")):
            raise ValueError("unrecognized observation source")
        key = (series, day.isoformat())
        if key in clean:
            raise ValueError("duplicate observation in capture")
        clean[key] = (value, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as source:
                existing = list(csv.DictReader(source))
        else:
            existing = []
        latest = {}
        for row in sorted(existing, key=lambda row: datetime.fromisoformat(row["available_at"])):
            latest[row["series"], row["observation_date"]] = row
        additions = []
        for (series, day), (value, url) in clean.items():
            previous = latest.get((series, day))
            if previous and float(previous["value"]) == value:
                continue
            if previous and datetime.fromisoformat(previous["available_at"]) >= stamp:
                raise ValueError("capture cannot overwrite or backdate an observed version")
            additions.append(dict(series=series, observation_date=day, available_at=stamp.isoformat(), value=value, source_url=url))
        if not additions:
            return 0
        with NamedTemporaryFile("w", dir=path.parent, newline="", encoding="utf-8", delete=False) as stream:
            temp = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows([*existing, *additions])
        try:
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        return len(additions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("GOLD_DRIVER_FLOWS_PATH") or "data_cache/gold_driver_flows.csv"))
    parser.add_argument("--cot-capture", type=Path, help="Previously captured official JSON response with observed timestamp")
    parser.add_argument("--wgc-excerpts", type=Path, help="Reviewed JSON list of monthly WGC excerpts")
    args = parser.parse_args()
    payload = json.loads(args.cot_capture.read_text()) if args.cot_capture else fetch_cot()
    cot = merge_observations(args.output, cot_observations(payload), captured_at=datetime.fromisoformat(payload["captured_at"]))
    wgc = 0
    if args.wgc_excerpts:
        excerpts = json.loads(args.wgc_excerpts.read_text())
        if any(x["series"] not in {"etf_net_tonnes", "central_bank_net_tonnes"} for x in excerpts):
            raise ValueError("only WGC ETF and central-bank series may be imported here")
        wgc = merge_observations(args.output, excerpts, captured_at=datetime.now(timezone.utc))
    print(json.dumps({"cot_versions_added": cot, "wgc_versions_added": wgc}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
