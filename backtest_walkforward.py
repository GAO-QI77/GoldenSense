"""Leakage-free walk-forward backtest for the GoldenSense T+1 gold signal.

Why this script exists
----------------------
``train_stacking.py`` selects features on the *entire* dataset before the
walk-forward split (look-ahead leakage) and only ever reports directional
accuracy (RMSE / MAE / ACC) -- never a cost-aware P&L. A high hit-rate does not
mean a profitable strategy. This harness answers the only question that matters
for production: *after realistic trading costs, does the signal have a positive,
stable Sharpe out-of-sample?*

What it does
------------
1. Builds features with the project's own FeatureEngineer (all transforms are
   causal / per-row, so building once introduces no leakage).
2. Runs an expanding-window walk-forward. Inside each fold it selects features
   and trains XGBoost on the training slice ONLY, then predicts the next block
   out-of-sample.
3. Concatenates the out-of-sample predictions and runs them through the tested
   ``backtest_engine`` under several position policies and cost levels, against
   a buy-and-hold benchmark.

XGBoost is used deliberately as the *best-case* single model: it is the
strongest L1 learner in the ensemble. If the strongest learner shows no edge on
clean data, the heavier GRU/Transformer stack will not rescue it.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from backtest_engine import run_backtest
from feature_engineer import FeatureEngineer

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover - dependency guard
    raise SystemExit(f"xgboost is required for this backtest: {exc}")


RAW_DATA_PATH = "raw_market_data.csv"
OUTPUT_DIR = "outputs"
TARGET_COL = "target_return_1d"


def build_feature_frame(raw_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Construct the causal feature frame + T+1 target from raw market data."""
    raw = pd.read_csv(raw_path, index_col=0)
    raw.index = pd.to_datetime(raw.index)

    fe = FeatureEngineer(horizons=[1])
    df = fe.preprocess(raw)
    df = fe.construct_seasonal_features(df)
    df = fe.construct_market_features(df)
    df = fe.adaptive_normalize(df)
    df = fe.construct_targets(df)
    return df


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if "target_" in c:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        cols.append(c)
    return cols


def _select_features_train_only(
    X_train: pd.DataFrame, y_train: pd.Series, top_k: int = 15
) -> List[str]:
    """Rank features on the training slice only (no look-ahead).

    Uses absolute Pearson correlation + RandomForest importance, mirroring the
    spirit of FeatureEngineer.select_features but fit strictly on train data.
    """
    from sklearn.ensemble import RandomForestRegressor

    corr = X_train.corrwith(y_train).abs().fillna(0.0)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train.values, y_train.values)
    imp = pd.Series(rf.feature_importances_, index=X_train.columns)
    imp = imp / (imp.max() or 1.0)
    corr_n = corr / (corr.max() or 1.0)
    score = (corr_n + imp) / 2.0
    return score.sort_values(ascending=False).head(top_k).index.tolist()


def walk_forward_predictions(
    df: pd.DataFrame,
    *,
    min_train_frac: float = 0.5,
    n_folds: int = 8,
    top_k: int = 15,
) -> pd.Series:
    """Return a Series of out-of-sample T+1 return predictions, indexed by date."""
    feature_cols = _feature_columns(df)
    X_all = df[feature_cols]
    y_all = df[TARGET_COL]

    n = len(df)
    min_train = int(n * min_train_frac)
    remaining = n - min_train
    test_size = max(10, remaining // n_folds)

    preds = pd.Series(index=df.index, dtype=float)
    start = min_train
    fold = 0
    while start < n:
        end = min(start + test_size, n)
        train_X = X_all.iloc[:start]
        train_y = y_all.iloc[:start]
        test_X = X_all.iloc[start:end]

        selected = _select_features_train_only(train_X, train_y, top_k=top_k)
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train_X[selected].values, train_y.values)
        preds.iloc[start:end] = model.predict(test_X[selected].values)

        fold += 1
        start = end

    return preds.dropna()


def positions_from_predictions(preds: pd.Series, policy: str) -> pd.Series:
    if policy == "always_in":
        return pd.Series(np.sign(preds.values), index=preds.index)
    if policy == "long_flat":
        return pd.Series((preds.values > 0).astype(float), index=preds.index)
    if policy == "threshold":
        thresh = np.median(np.abs(preds.values))
        pos = np.where(np.abs(preds.values) > thresh, np.sign(preds.values), 0.0)
        return pd.Series(pos, index=preds.index)
    raise ValueError(f"unknown policy: {policy}")


def directional_accuracy(preds: pd.Series, realized: pd.Series) -> float:
    aligned = realized.loc[preds.index]
    return float(((preds.values > 0) == (aligned.values > 0)).mean())


def run() -> Dict:
    df = build_feature_frame()
    realized = df[TARGET_COL]  # realized T+1 return == the P&L of a 1-day hold
    preds = walk_forward_predictions(df)
    oos_realized = realized.loc[preds.index]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    acc = directional_accuracy(preds, realized)
    span = (preds.index.min().date(), preds.index.max().date())

    cost_levels = [0.0, 2.0, 5.0, 10.0]
    policies = ["always_in", "long_flat", "threshold"]

    # Buy-and-hold benchmark (always long, no turnover after entry).
    bh = run_backtest(
        forward_returns=oos_realized,
        positions=pd.Series(1.0, index=preds.index),
        cost_bps=0.0,
    )

    rows: List[Dict] = []
    for policy in policies:
        positions = positions_from_predictions(preds, policy)
        for cost in cost_levels:
            res = run_backtest(
                forward_returns=oos_realized, positions=positions, cost_bps=cost
            )
            rows.append(
                {
                    "policy": policy,
                    "cost_bps": cost,
                    "sharpe_net": round(res.sharpe, 3),
                    "sortino": round(res.sortino, 3),
                    "ann_return": round(res.ann_return, 4),
                    "max_drawdown": round(res.max_drawdown, 4),
                    "calmar": round(res.calmar, 3),
                    "hit_rate": round(res.hit_rate, 4),
                    "avg_turnover": round(res.avg_turnover, 4),
                    "num_trades": res.num_trades,
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUTPUT_DIR, "walkforward_backtest.csv"), index=False)

    summary = {
        "oos_samples": int(len(preds)),
        "oos_span": [str(span[0]), str(span[1])],
        "directional_accuracy": round(acc, 4),
        "buy_and_hold": {
            "sharpe": round(bh.sharpe, 3),
            "ann_return": round(bh.ann_return, 4),
            "max_drawdown": round(bh.max_drawdown, 4),
            "total_return": round(bh.total_return, 4),
        },
        "results": rows,
    }
    with open(os.path.join(OUTPUT_DIR, "walkforward_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _print_report(summary, table)
    return summary


def _print_report(summary: Dict, table: pd.DataFrame) -> None:
    print("=" * 72)
    print("GoldenSense T+1 walk-forward backtest (leakage-free, cost-aware)")
    print("=" * 72)
    print(
        f"Out-of-sample window : {summary['oos_span'][0]} -> {summary['oos_span'][1]} "
        f"({summary['oos_samples']} trading days)"
    )
    print(f"Directional accuracy : {summary['directional_accuracy']:.2%}  "
          "(coin flip = 50%)")
    bh = summary["buy_and_hold"]
    print(
        f"Buy & hold benchmark : Sharpe {bh['sharpe']:.2f} | "
        f"AnnRet {bh['ann_return']:.2%} | MaxDD {bh['max_drawdown']:.2%}"
    )
    print("-" * 72)
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(table.to_string(index=False))
    print("-" * 72)
    print("Read: a strategy is only worth deploying if net Sharpe stays clearly")
    print("positive (>~1) at a realistic cost (XAUUSD retail ~2-5 bps/side).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
