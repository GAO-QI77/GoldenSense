"""Advanced, literature-grounded gold strategies, evaluated through the same
tested cost-aware backtest engine.

Methods (canonical institutional approaches; parameters fixed a priori):
- Multi-timescale trend ensemble -- Hurst, Ooi & Pedersen, "A Century of
  Evidence on Trend-Following Investing" (AQR, 2017). Combine 1/3/12-month
  trend signals instead of a single lookback to reduce parameter fragility.
- Volatility-managed exposure -- Moreira & Muir, "Volatility-Managed
  Portfolios" (Journal of Finance, 2017). Scale exposure down when realized
  volatility is high.
- Rebalancing band -- standard transaction-cost control: only trade when the
  target position moves materially, so vol-management does not churn.

NOTE: parameters (1/3/12-month lookbacks, 10% vol target, 1.5x cap, 0.1 band)
are chosen a priori from the literature, NOT fit to this data. The evaluation is
therefore out-of-sample by construction. We test one small pre-registered set
and report it -- no "try many, keep the best" data-snooping.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest_engine import run_backtest
from strategy_factors import forward_returns_1d, trend_signal, vol_target_position


# --------------------------------------------------------------------------- #
# New causal primitives (unit-tested in tests/test_strategy_advanced.py)
# --------------------------------------------------------------------------- #
def multi_trend_signal(
    prices: pd.Series, lookbacks: Tuple[int, ...] = (21, 63, 252)
) -> pd.Series:
    """Average of long/flat trend sub-signals across several lookbacks.

    Output is graded exposure in [0, 1]: e.g. 0, 1/3, 2/3, 1 for three lookbacks.
    """
    sub = [trend_signal(prices, lb) for lb in lookbacks]
    stacked = pd.concat(sub, axis=1)
    return stacked.mean(axis=1)


def rebalance_band(target: pd.Series, band: float) -> pd.Series:
    """Hold the current position until the target moves by more than ``band``.

    Causal: each output depends only on the running position and the current
    target. This is what keeps vol-managed strategies from churning.
    """
    out: List[float] = []
    current = 0.0
    for v in target.values:
        if abs(float(v) - current) > band:
            current = float(v)
        out.append(current)
    return pd.Series(out, index=target.index)


# --------------------------------------------------------------------------- #
# Strategy set
# --------------------------------------------------------------------------- #
def _align(positions: pd.Series, fwd: pd.Series) -> pd.Series:
    return positions.reindex(fwd.index).fillna(0.0)


def build_strategies(raw: pd.DataFrame) -> Dict[str, pd.Series]:
    gold = raw["Gold"].astype(float)
    fwd = forward_returns_1d(gold)
    gold_ret = gold.pct_change().fillna(0.0)

    strategies: Dict[str, pd.Series] = {}
    strategies["buy_and_hold"] = _align(pd.Series(1.0, index=gold.index), fwd)

    # Stage-B champion for reference.
    strategies["trend_200d_long_flat"] = _align(trend_signal(gold, 200), fwd)

    # A1: multi-timescale trend ensemble (1/3/12 months), graded long/flat.
    multi = multi_trend_signal(gold, lookbacks=(21, 63, 252))
    strategies["multi_trend_1_3_12m"] = _align(multi, fwd)

    # A2: vol-managed multi-trend (10% target, cap 1.5x) + rebalancing band.
    vt = vol_target_position(
        multi, gold_ret, target_vol=0.10, lookback=63, max_leverage=1.5
    )
    vt_banded = rebalance_band(vt, band=0.10)
    strategies["multi_trend_volmgd_banded"] = _align(vt_banded, fwd)

    # A3: Moreira-Muir style vol-managed long (inverse-variance scaled long).
    realized_var = (gold_ret.rolling(63).std() ** 2) * 252
    target_var = 0.10 ** 2
    mm = (target_var / realized_var).replace([np.inf, -np.inf], np.nan).clip(0.0, 1.5)
    mm = rebalance_band(mm.fillna(0.0), band=0.10)
    strategies["volmgd_long_moreira_muir"] = _align(mm, fwd)

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
    table.to_csv("outputs/advanced_backtest.csv", index=False)
    with open("outputs/advanced_backtest.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    span = (fwd.index.min().date(), fwd.index.max().date())
    print("=" * 88)
    print("GoldenSense Stage C -- literature-grounded strategies (full-sample, cost-aware)")
    print("=" * 88)
    print(f"Sample: {span[0]} -> {span[1]} ({len(fwd)} days). A-priori params, no fitting -> OOS.")
    print("-" * 88)
    view = table[table["cost_bps"] == 2.0].sort_values("calmar", ascending=False)
    with pd.option_context("display.width", 150, "display.max_columns", None):
        print(view.to_string(index=False))
    print("-" * 88)
    print("Deploy bar: net Sharpe > ~1 AND beats buy_and_hold on Sharpe/Calmar,")
    print("or matches return at materially smaller drawdown.")
    return table


if __name__ == "__main__":
    run()
