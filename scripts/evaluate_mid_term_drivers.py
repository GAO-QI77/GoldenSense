"""Run same-sample middle-term prediction diagnostics on the local dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_sources import load_market_data
from strategy_macro import walk_forward_drivers


def _paired_mse_difference(rows: pd.DataFrame, horizon: int, candidate: str, baseline: str) -> dict:
    realized = rows["realized_return"]
    difference = rows[f"{candidate}_forecast"].sub(realized).pow(2).sub(
        rows[f"{baseline}_forecast"].sub(realized).pow(2)
    )
    interval = None
    if len(difference) >= 2 * horizon:
        from statsmodels.api import OLS

        fitted = OLS(difference.to_numpy(), np.ones((len(difference), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": horizon - 1},
        )
        interval = fitted.conf_int()[0].tolist()
    return {"paired_n": len(difference), "mean_difference": float(difference.mean()), "ci95_hac": interval}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", required=True, help="Last completed bar date")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--market-data", help="Alternative market data CSV with Gold and driver columns")
    parser.add_argument("--evaluation-from", help="First eligible evaluation date (YYYY-MM-DD)")
    parser.add_argument("--oil-interaction", action="store_true", help="Compare WTI and real-rate interaction candidates")
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
    models = ("zero", "trend", "driver")
    if args.oil_interaction:
        models = (*models, "oil_additive", "oil_interaction")
        summary["oil_proxy"] = "CL=F"
        summary["point_in_time_frame_sha256"] = hashlib.sha256(raw.to_csv().encode()).hexdigest()
    for horizon in (21, 63):
        rows = walk_forward_drivers(raw, horizon=horizon, include_oil_interaction=args.oil_interaction)
        if first is not None:
            rows = rows.loc[first:]
        if rows.empty:
            raise ValueError("no mature forecasts in the selected evaluation period")
        losses = {
            name: float(rows[f"{name}_forecast"].sub(rows["realized_return"]).pow(2).mean())
            for name in models
        }
        rows.to_csv(args.output_dir / f"window_{horizon}.csv", index_label="decision_date")
        summary["windows"][horizon] = {
            "drivers": rows.attrs["drivers"],
            "mature_forecasts": len(rows),
            "mse": losses,
        }
        if args.oil_interaction:
            summary["windows"][horizon].update(
                oil_features=rows.attrs["oil_features"],
                additive_vs_driver=_paired_mse_difference(rows, horizon, "oil_additive", "driver"),
                interaction_vs_additive=_paired_mse_difference(rows, horizon, "oil_interaction", "oil_additive"),
                interaction_vs_zero=_paired_mse_difference(rows, horizon, "oil_interaction", "zero"),
            )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
