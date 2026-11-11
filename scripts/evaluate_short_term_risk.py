"""Compare variance baselines on one instrument's daily close history."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gold_sessions import completed_gold_sessions
from risk_review import GARCH_CANDIDATE
from vol_models import walk_forward_variance


def completed_sessions(prices: pd.Series, calendar: pd.DataFrame, *, through: str) -> pd.Series:
    return completed_gold_sessions(prices, calendar, through=through)["Gold"].rename(prices.name)


def _paired_block_interval(errors: pd.Series, block: int = 21) -> list[float] | None:
    if len(errors) < block * 2:
        return None
    rng = np.random.default_rng(20261110)
    blocks_needed = (len(errors) + block - 1) // block
    starts = rng.integers(0, len(errors), size=(1000, blocks_needed))
    indices = ((starts[:, :, None] + np.arange(block)) % len(errors)).reshape(len(starts), -1)
    draws = errors.to_numpy()[indices[:, :len(errors)]].mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def _finite_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def summarize_risk(
    prices: pd.Series, *, horizon: int, min_history: int = 260,
    evaluation_from: pd.Timestamp | None = None, include_challengers: bool = False,
    garch_variance: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    comparison = walk_forward_variance(
        prices, horizon=horizon, min_history=min_history, include_challengers=include_challengers,
    )
    challengers = ("log_har", "log_har_vra") if include_challengers else ()
    if garch_variance is not None:
        comparison[f"{GARCH_CANDIDATE}_variance"] = garch_variance.reindex(comparison.index)
        challengers = (*challengers, GARCH_CANDIDATE)
    models = ("rolling", "ewma", "har", *challengers)
    forward = prices.shift(-horizon).div(prices).sub(1).reindex(comparison.index)
    comparison["realized_return"] = forward
    for model in models:
        variance = comparison[f"{model}_variance"]
        comparison[f"{model}_qlike"] = np.log(variance) + comparison["realized_variance"] / variance
        scale = np.sqrt(variance)
        # A standardized return produced at origin i matures at i+horizon.
        matured = forward.div(scale).shift(horizon).rolling(500, min_periods=126)
        comparison[f"{model}_p10"] = matured.quantile(0.10) * scale
        comparison[f"{model}_p90"] = matured.quantile(0.90) * scale

    if evaluation_from is not None:
        comparison = comparison.loc[evaluation_from:]
    if comparison.empty:
        raise ValueError("no mature forecasts in the evaluation period")
    losses = {}
    bands = {}
    for model in models:
        losses[model] = _finite_number(comparison[f"{model}_qlike"].mean())
        valid = comparison.dropna(subset=[f"{model}_p10", f"{model}_p90"])
        standardized = comparison["realized_return"] / np.sqrt(comparison[f"{model}_variance"])
        result = {
            "n": len(valid),
            "standardized_bias": _finite_number(standardized.mean()),
            "standardized_dispersion": _finite_number(standardized.std(ddof=1)),
        }
        if not valid.empty:
            below = valid["realized_return"] < valid[f"{model}_p10"]
            above = valid["realized_return"] > valid[f"{model}_p90"]
            result.update(
                coverage=float((~(below | above)).mean()),
                below=float(below.mean()),
                above=float(above.mean()),
                mean_width=float((valid[f"{model}_p90"] - valid[f"{model}_p10"]).mean()),
            )
        bands[model] = result

    difference = comparison["har_qlike"] - comparison["ewma_qlike"]
    paired = {}
    for model in challengers:
        candidate = (comparison[f"{model}_qlike"] - comparison["ewma_qlike"]).dropna()
        paired[model] = {
            "mean_difference": _finite_number(candidate.mean()),
            "n": len(candidate),
            "ci95": _paired_block_interval(candidate) if len(candidate) == len(comparison) else None,
        }
    return comparison, {
        "mature_forecasts": len(comparison),
        "first_origin": str(comparison.index[0].date()),
        "last_origin": str(comparison.index[-1].date()),
        "mean_qlike": losses,
        "har_floor_hits": int(comparison["har_variance"].le(horizon * 1e-12).sum()),
        "har_minus_ewma": float(difference.mean()),
        "har_minus_ewma_block_ci95": _paired_block_interval(difference),
        "bands": bands,
        "challenger_qlike_vs_ewma": paired,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("closes", type=Path, help="CSV with Date and Gold columns")
    parser.add_argument("--instrument", required=True, help="Source instrument identifier")
    parser.add_argument("--price-column", default="Gold", help="Close column (default Gold)")
    parser.add_argument("--through", required=True, help="Date cutoff, or offset timestamp with --sessions")
    parser.add_argument("--sessions", type=Path, help="Verified session CSV with Date,close_at,available_at")
    parser.add_argument("--evaluation-from", help="First eligible evaluation origin (YYYY-MM-DD)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--challengers", action="store_true", help="Include log-HAR and mature-error VRA")
    parser.add_argument("--student-t-garch", action="store_true", help="Also fit rolling Student-t GARCH(1,1)")
    args = parser.parse_args()
    include_challengers = args.challengers or args.student_t_garch

    source_bytes = args.closes.read_bytes()
    frame = pd.read_csv(io.BytesIO(source_bytes), usecols=["Date", args.price_column])
    dates = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="raise")
    prices = pd.Series(frame[args.price_column].to_numpy(), index=pd.DatetimeIndex(dates))
    session_bytes = args.sessions.read_bytes() if args.sessions else None
    if session_bytes is not None:
        calendar = pd.read_csv(io.BytesIO(session_bytes))
        prices = completed_sessions(prices, calendar, through=args.through)
    else:
        cutoff = pd.to_datetime(args.through, format="%Y-%m-%d")
        prices = prices.loc[prices.index <= cutoff]
    if prices.empty:
        parser.error("no prices on or before --through")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    version = 4 if args.student_t_garch else 3 if include_challengers else 2
    summary = {
        "research_horizon": "short_term",
        "protocol": f"daily_close_variance_baselines_v{version}",
        "calendar_sha256": hashlib.sha256(session_bytes).hexdigest() if session_bytes else None,
        "evaluation_from": args.evaluation_from,
        "paired_bootstrap_block": 21,
        "paired_bootstrap_sampling": "circular_moving_block",
        "band_min_mature_samples": 126,
        "band_calibration_window": 500,
        "instrument": args.instrument,
        "source_file": args.closes.name,
        "price_column": args.price_column,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "model_code_sha256": hashlib.sha256(
            (Path(__file__).resolve().parent.parent / "vol_models.py").read_bytes()
        ).hexdigest(),
        "through": args.through,
        "price_start": str(prices.index[0].date()),
        "price_end": str(prices.index[-1].date()),
        "min_training_returns": 260,
        "rolling_window": 22,
        "ewma_decay": 0.94,
        "evidence_status": "exploratory",
        "windows": {},
    }
    evaluation_from = pd.to_datetime(args.evaluation_from, format="%Y-%m-%d")
    horizons = (1, 5, 21)
    garch = None
    if args.student_t_garch:
        import arch
        from garch_risk import FIT_WINDOW, REFIT_EVERY, walk_forward_garch
        from risk_review import garch_model_hash

        garch = walk_forward_garch(prices, horizons=horizons)
        summary["garch"] = {
            "arch_version": arch.__version__,
            "fit_window": FIT_WINDOW,
            "refit_every": REFIT_EVERY,
            "code_sha256": garch_model_hash(),
        }
    for horizon in horizons:
        comparison, diagnostics = summarize_risk(
            prices, horizon=horizon, evaluation_from=evaluation_from, include_challengers=include_challengers,
            garch_variance=garch[horizon] if garch is not None else None,
        )
        comparison.to_csv(out / f"window_{horizon}.csv")
        if include_challengers:
            review = {}
            for model, difference in diagnostics["challenger_qlike_vs_ewma"].items():
                bands = diagnostics["bands"][model]
                blockers = []
                if session_bytes is None:
                    blockers.append("session_timestamps_missing")
                if evaluation_from is None:
                    blockers.append("evaluation_start_missing")
                if difference["n"] < diagnostics["mature_forecasts"]:
                    blockers.append("incomplete_forecasts")
                if bands["n"] < 126:
                    blockers.append("insufficient_mature_bands")
                if not 0.75 <= bands.get("coverage", 0) <= 0.85 or max(bands.get("below", 1), bands.get("above", 1)) > 0.15:
                    blockers.append("coverage_or_tail_failed")
                if difference["ci95"] is None or difference["ci95"][1] >= 0:
                    blockers.append("incremental_error_not_demonstrated")
                review[model] = {"status": "hold" if blockers else "review_required", "blockers": blockers}
            diagnostics["release_review"] = review
        summary["windows"][horizon] = diagnostics
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    report = json.dumps(summary, indent=2, allow_nan=False)
    (out / "summary.json").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
