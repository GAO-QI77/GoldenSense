"""Mid-term macro factor sleeve for gold (weeks-to-months horizon).

Adds the first-principles drivers that pure price trend ignores, each as a
fixed-rule causal signal (a-priori parameters, nothing fitted -- same
discipline as strategy_factors / strategy_advanced):

- F1 real-rate momentum      : long gold while the 10Y TIPS real yield is
                               falling (63d change < 0). The canonical driver.
- F2 USD trend               : long gold while the dollar index sits below its
                               50d mean (validated as S4 in Stage B).
- F3 inflation-expectation   : long gold while the 10Y breakeven is rising
  momentum                     (63d change > 0) -- inflation-hedge demand.
- F4 flow confirmation proxy : price uptrend confirmed by expanding GLD dollar
                               volume (21d mean above 63d mean). Explicitly a
                               *proxy*; true ETF holdings need a paid source.

The composite is the equal-weight mean of the available factor signals, i.e.
graded exposure in [0, 1], optionally vol-targeted. Missing columns simply
drop out and are reported, so the sleeve degrades honestly on the base CSV.

Roadmap (needs data we do not have for free): futures term-structure carry,
COT positioning extremes, central-bank purchase flow.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest_engine import run_backtest
from strategy_factors import forward_returns_1d, trend_signal, vol_target_position


# --------------------------------------------------------------------------- #
# Causal factor primitives
# --------------------------------------------------------------------------- #
def falling_signal(series: pd.Series, lookback: int = 63) -> pd.Series:
    """1.0 while the series has fallen over the trailing ``lookback`` days."""
    change = series.astype(float).diff(lookback)
    sig = (change < 0).astype(float)
    sig[change.isna()] = 0.0
    return sig


def rising_signal(series: pd.Series, lookback: int = 63) -> pd.Series:
    """1.0 while the series has risen over the trailing ``lookback`` days."""
    change = series.astype(float).diff(lookback)
    sig = (change > 0).astype(float)
    sig[change.isna()] = 0.0
    return sig


def below_ma_signal(series: pd.Series, lookback: int = 50) -> pd.Series:
    """1.0 while the series sits below its trailing moving average."""
    ma = series.astype(float).rolling(lookback).mean()
    sig = (series < ma).astype(float)
    sig[ma.isna()] = 0.0
    return sig


def flow_confirmation_signal(
    prices: pd.Series,
    dollar_volume: pd.Series,
    *,
    trend_lookback: int = 50,
    fast: int = 21,
    slow: int = 63,
) -> pd.Series:
    """Price uptrend AND expanding participation (fast vol-flow above slow)."""
    uptrend = trend_signal(prices, trend_lookback)
    dv = dollar_volume.astype(float)
    fast_mean = dv.rolling(fast).mean()
    slow_mean = dv.rolling(slow).mean()
    expanding = (fast_mean > slow_mean).astype(float)
    expanding[fast_mean.isna() | slow_mean.isna()] = 0.0
    return uptrend * expanding


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
def build_factor_signals(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return (per-factor signal frame, list of factors actually available)."""
    gold = raw["Gold"].astype(float)
    signals: Dict[str, pd.Series] = {}

    if "Real_10Y" in raw.columns:
        signals["real_rate_momentum"] = falling_signal(raw["Real_10Y"], 63)
    if "USD_Index" in raw.columns:
        signals["usd_downtrend"] = below_ma_signal(raw["USD_Index"], 50)
    if "Breakeven_10Y" in raw.columns:
        signals["inflation_expectation"] = rising_signal(raw["Breakeven_10Y"], 63)
    if "GLD_DollarVolume" in raw.columns:
        signals["flow_confirmation"] = flow_confirmation_signal(
            gold, raw["GLD_DollarVolume"]
        )

    if not signals:
        # Base CSV without macro columns: fall back to the trend anchor so the
        # sleeve stays defined (and mark it via the factor list).
        signals["price_trend_fallback"] = trend_signal(gold, 200)

    frame = pd.DataFrame(signals)
    return frame, list(frame.columns)


def macro_composite_position(raw: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    """Equal-weight mean of the available factor signals -> exposure in [0,1]."""
    frame, used = build_factor_signals(raw)
    return frame.mean(axis=1), used


def build_strategies(raw: pd.DataFrame) -> Dict[str, pd.Series]:
    gold = raw["Gold"].astype(float)
    fwd = forward_returns_1d(gold)
    gold_ret = gold.pct_change().fillna(0.0)

    def _align(p: pd.Series) -> pd.Series:
        return p.reindex(fwd.index).fillna(0.0)

    strategies: Dict[str, pd.Series] = {}
    strategies["buy_and_hold"] = _align(pd.Series(1.0, index=gold.index))
    strategies["trend_200d_long_flat"] = _align(trend_signal(gold, 200))

    composite, used = macro_composite_position(raw)
    strategies[f"macro_composite_{len(used)}f"] = _align(composite)

    # Composite gated by price trend: only deploy macro exposure in uptrends.
    gated = composite * trend_signal(gold, 200)
    strategies["macro_composite_trend_gated"] = _align(gated)

    # Vol-targeted version of the gated sleeve (10% target, 1.5x cap).
    vt = vol_target_position(
        gated, gold_ret, target_vol=0.10, lookback=63, max_leverage=1.5
    )
    strategies["macro_gated_vol_target_10pct"] = _align(vt)

    return strategies


def run(raw_path: str = None) -> pd.DataFrame:
    import json
    import os

    from data_sources import load_market_data

    if raw_path:
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True).ffill().dropna()
        source = raw_path
    else:
        raw, source = load_market_data()

    gold = raw["Gold"].astype(float)
    fwd = forward_returns_1d(gold)
    strategies = build_strategies(raw)
    _, used = macro_composite_position(raw)

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
    table.to_csv("outputs/macro_backtest.csv", index=False)
    with open("outputs/macro_backtest.json", "w") as f:
        json.dump({"factors_used": used, "data_source": source, "results": rows}, f,
                  indent=2, ensure_ascii=False)

    span = (fwd.index.min().date(), fwd.index.max().date())
    print("=" * 88)
    print("GoldenSense Stage D -- macro factor sleeve (full-sample, cost-aware)")
    print("=" * 88)
    print(f"Sample: {span[0]} -> {span[1]} ({len(fwd)} days) | data={source} | factors={used}")
    print("-" * 88)
    view = table[table["cost_bps"] == 2.0].sort_values("calmar", ascending=False)
    with pd.option_context("display.width", 150, "display.max_columns", None):
        print(view.to_string(index=False))
    return table


if __name__ == "__main__":
    run()
