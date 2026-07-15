"""Rules-based factor strategies for gold, evaluated through the tested
cost-aware backtest engine.

Design philosophy (the lesson from Stage A)
-------------------------------------------
A fitted daily-direction model showed *negative* edge out-of-sample. So we do
NOT fit anything here. These are fixed-rule strategies whose parameters are
chosen a priori from the cross-asset momentum / trend-following literature
(lookbacks of roughly 1-12 months). Because nothing is fit to this data, the
result is out-of-sample by construction -- there is no leakage to fix.

The bar for "worth deploying":
- Net Sharpe (after 2-5 bps/side) clearly positive AND
- Beats buy-and-hold on a *risk-adjusted* basis (higher Sharpe / Calmar or
  materially smaller drawdown). Matching buy-and-hold's return with less than
  half the drawdown is a real, deployable improvement for a research tool.

All primitives are causal: a value at time t uses only data at or before t.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from backtest_engine import run_backtest


# --------------------------------------------------------------------------- #
# Causal signal primitives (unit-tested in tests/test_strategy_factors.py)
# --------------------------------------------------------------------------- #
def forward_returns_1d(prices: pd.Series) -> pd.Series:
    """Return earned by holding from each date's close to the next close."""
    return (prices.shift(-1) / prices - 1.0).dropna()


def trend_signal(prices: pd.Series, lookback: int) -> pd.Series:
    """1.0 when price is above its trailing moving average, else 0.0."""
    ma = prices.rolling(lookback).mean()
    sig = (prices > ma).astype(float)
    sig[ma.isna()] = 0.0
    return sig


def momentum_signal(prices: pd.Series, lookback: int) -> pd.Series:
    """Sign of the trailing ``lookback``-day return."""
    trailing = prices / prices.shift(lookback) - 1.0
    return pd.Series(np.sign(trailing.fillna(0.0).values), index=prices.index)


def vol_target_position(
    base_position: pd.Series,
    asset_returns: pd.Series,
    *,
    target_vol: float,
    lookback: int,
    max_leverage: float,
) -> pd.Series:
    """Scale a base position so realized vol targets ``target_vol`` (annualized).

    Uses trailing realized vol (known at t), so there is no look-ahead.
    """
    realized_vol = asset_returns.rolling(lookback).std() * np.sqrt(252)
    scale = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan)
    scale = scale.clip(lower=0.0, upper=max_leverage).fillna(0.0)
    return base_position * scale


# --------------------------------------------------------------------------- #
# Strategy definitions over the full sample
# --------------------------------------------------------------------------- #
def _align(positions: pd.Series, fwd: pd.Series) -> pd.Series:
    return positions.reindex(fwd.index).fillna(0.0)


def build_strategies(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return {name: position series aligned to forward-return dates}."""
    gold = df["Gold"].astype(float)
    fwd = forward_returns_1d(gold)
    gold_ret = gold.pct_change().fillna(0.0)

    strategies: Dict[str, pd.Series] = {}

    # Benchmark: always long.
    strategies["buy_and_hold"] = _align(pd.Series(1.0, index=gold.index), fwd)

    # S1: 200-day trend filter, long/flat (classic Faber-style timing).
    trend200 = trend_signal(gold, lookback=200)
    strategies["trend_200d_long_flat"] = _align(trend200, fwd)

    # S2: 50-day trend filter, faster.
    trend50 = trend_signal(gold, lookback=50)
    strategies["trend_50d_long_flat"] = _align(trend50, fwd)

    # S3: 3-month time-series momentum, long/short.
    mom63 = momentum_signal(gold, lookback=63)
    strategies["tsmom_3m_long_short"] = _align(mom63, fwd)

    # S4: macro filter -- long gold only when USD index is in a downtrend.
    if "USD_Index" in df.columns:
        usd = df["USD_Index"].astype(float)
        usd_downtrend = (usd < usd.rolling(50).mean()).astype(float)
        usd_downtrend[usd.rolling(50).mean().isna()] = 0.0
        macro = (trend200.values * usd_downtrend.values)
        strategies["trend200_and_weak_usd"] = _align(
            pd.Series(macro, index=gold.index), fwd
        )

    # S5: vol-targeted 200-day trend (10% annualized target, cap 1.5x).
    vt = vol_target_position(
        trend200, gold_ret, target_vol=0.10, lookback=20, max_leverage=1.5
    )
    strategies["trend_200d_vol_target_10pct"] = _align(vt, fwd)

    return strategies


def run(raw_path: str = None) -> pd.DataFrame:
    import os
    import json

    if raw_path:
        raw = pd.read_csv(raw_path, index_col=0)
        raw.index = pd.to_datetime(raw.index)
        raw = raw.ffill().dropna()
    else:
        from data_sources import load_market_data

        raw, _source = load_market_data()

    gold = raw["Gold"].astype(float)
    fwd = forward_returns_1d(gold)
    strategies = build_strategies(raw)

    cost_levels = [0.0, 2.0, 5.0]
    rows: List[Dict] = []
    for name, positions in strategies.items():
        for cost in cost_levels:
            res = run_backtest(forward_returns=fwd, positions=positions, cost_bps=cost)
            rows.append(
                {
                    "strategy": name,
                    "cost_bps": cost,
                    "sharpe_net": round(res.sharpe, 3),
                    "sortino": round(res.sortino, 3),
                    "ann_return": round(res.ann_return, 4),
                    "max_drawdown": round(res.max_drawdown, 4),
                    "calmar": round(res.calmar, 3),
                    "avg_turnover": round(res.avg_turnover, 4),
                    "num_trades": res.num_trades,
                }
            )

    table = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    table.to_csv("outputs/factor_backtest.csv", index=False)
    with open("outputs/factor_backtest.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    span = (fwd.index.min().date(), fwd.index.max().date())
    print("=" * 84)
    print("GoldenSense Stage B -- rules-based factor strategies (full-sample, cost-aware)")
    print("=" * 84)
    print(f"Sample: {span[0]} -> {span[1]} ({len(fwd)} trading days). No fitting -> OOS by construction.")
    print("-" * 84)
    # Show the realistic 2 bps/side view, sorted by risk-adjusted return.
    view = table[table["cost_bps"] == 2.0].sort_values("sharpe_net", ascending=False)
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(view.to_string(index=False))
    print("-" * 84)
    print("Deploy bar: net Sharpe > ~1 AND beats buy_and_hold on Sharpe/Calmar or")
    print("delivers similar return at materially smaller drawdown.")
    return table


if __name__ == "__main__":
    run()
