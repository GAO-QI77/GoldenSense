"""Build the point-in-time macro archive from ALFRED real-time periods."""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo
from zipfile import BadZipFile, ZipFile

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_sources import FRED_SERIES

ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
NY = ZoneInfo("America/New_York")
DEFAULT_PATH = "data_cache/macro_vintages.csv"


def _record_version(rows, series: str, observed: str, vintage: str, raw_value) -> None:
    if raw_value in (None, "", "."):
        return
    value = float(raw_value)
    day = date.fromisoformat(observed)
    as_of = date.fromisoformat(vintage)
    if not math.isfinite(value) or as_of < day:
        raise ValueError(f"{series}: invalid version")
    # Wait through the entire New York vintage day when the clock is unknown.
    available = datetime.combine(as_of + timedelta(days=1), time.min, NY)
    key = (series, day.isoformat(), available.astimezone(timezone.utc).isoformat())
    if key in rows and rows[key] != value:
        raise ValueError(f"{series}: conflicting version")
    rows[key] = value


def _write_archive(path: Path, rows) -> dict[str, int]:
    counts = dict.fromkeys(FRED_SERIES.values(), 0)
    for series, _, _ in rows:
        counts[series] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError("at least one required series has no numeric versions")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
            temp = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(["series_id", "observation_date", "available_at", "value"])
            writer.writerows((*key, value) for key, value in sorted(rows.items()))
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return counts


def refresh_archive(path: Path, fetch) -> dict[str, int]:
    rows = {}
    for series in FRED_SERIES.values():
        offset = 0
        while True:
            payload = fetch({
                "series_id": series, "file_type": "json", "output_type": 1,
                "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
                "observation_start": "2004-01-01", "limit": 10000, "offset": offset,
            })
            observations = payload["observations"]
            total = int(payload["count"])
            if offset + len(observations) > total or (offset < total and not observations):
                raise ValueError(f"{series}: incomplete download")
            for obs in observations:
                _record_version(rows, series, obs["date"], obs["realtime_start"], obs["value"])
            offset += len(observations)
            if offset == total:
                break
    return _write_archive(path, rows)


def import_exports(path: Path, source: Path) -> dict[str, int]:
    """Import the public ALFRED real-time period ZIPs listed in a checksum manifest."""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    rows = {}
    for series in FRED_SERIES.values():
        for batch in manifest[series]["batches"]:
            name = batch["file"]
            if Path(name).name != name:
                raise ValueError("export manifest must use a local filename")
            content = (source / name).read_bytes()
            if sha256(content).hexdigest() != batch["sha256"]:
                raise ValueError(f"{series}: export checksum mismatch")
            with ZipFile(BytesIO(content)) as export:
                tables = [entry for entry in export.namelist() if entry.endswith(".csv")]
                if len(tables) != 1:
                    raise ValueError(f"{series}: expected one observation table")
                reader = csv.DictReader(StringIO(export.read(tables[0]).decode("utf-8-sig")))
                fields = {"period_start_date", "realtime_start_date", series}
                if not fields.issubset(reader.fieldnames or ()):
                    raise ValueError(f"{series}: unexpected export columns")
                count = 0
                for row in reader:
                    _record_version(rows, series, row["period_start_date"], row["realtime_start_date"], row[series])
                    count += 1
                if count != batch["rows"]:
                    raise ValueError(f"{series}: export row count mismatch")
    return _write_archive(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("MACRO_VINTAGES_PATH", DEFAULT_PATH)))
    parser.add_argument("--alfred-exports", type=Path, help="Directory of public ALFRED ZIPs and manifest")
    args = parser.parse_args()
    key = os.environ.get("FRED_API_KEY")
    if not key and not args.alfred_exports:
        parser.error("FRED_API_KEY or --alfred-exports is required")

    try:
        if args.alfred_exports:
            counts = import_exports(args.output, args.alfred_exports)
        else:
            with requests.Session() as session:
                def fetch(params):
                    response = session.get(ENDPOINT, params={**params, "api_key": key}, timeout=30)
                    response.raise_for_status()
                    return response.json()

                counts = refresh_archive(args.output, fetch)
    except (requests.RequestException, ValueError, TypeError, KeyError, OSError, BadZipFile) as exc:
        # HTTP exceptions can include the URL with the credential.
        print(f"Macro archive refresh failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
