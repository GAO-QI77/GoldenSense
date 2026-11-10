"""Run same-sample middle-term prediction diagnostics on the local dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_sources import load_market_data
from strategy_macro import walk_forward_drivers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", required=True, help="Last completed bar date")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--market-data", help="Alternative market data CSV with Gold and driver columns")
    parser.add_argument("--evaluation-from", help="First eligible evaluation date (YYYY-MM-DD)")
    args = parser.parse_args()

    options = {"extended_path": args.market_data} if args.market_data else {}
    raw, source = load_market_data(require_macro_vintages=True, **options)
    if args.market_data and source != "extended":
        parser.error("market data file is unavailable")
    first = pd.to_datetime(args.evaluation_from, format="%Y-%m-%d")
    raw = raw.loc[:pd.to_datetime(args.through, format="%Y-%m-%d")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "research_horizon": "mid_term",
        "evidence_status": "exploratory",
        "source": source,
        "through": args.through,
        "evaluation_from": args.evaluation_from,
        "target": "forward_log_return",
        "windows": {},
    }
    for horizon in (21, 63):
        rows = walk_forward_drivers(raw, horizon=horizon)
        if first is not None:
            rows = rows.loc[first:]
        if rows.empty:
            raise ValueError("no mature forecasts in the selected evaluation period")
        losses = {
            name: float(rows[f"{name}_forecast"].sub(rows["realized_return"]).pow(2).mean())
            for name in ("zero", "trend", "driver")
        }
        rows.to_csv(args.output_dir / f"window_{horizon}.csv", index_label="decision_date")
        summary["windows"][horizon] = {
            "drivers": rows.attrs["drivers"],
            "mature_forecasts": len(rows),
            "mse": losses,
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
