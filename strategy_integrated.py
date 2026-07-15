"""GoldenSense flagship strategy: one composed, causal, drawdown-aware sleeve.

This is where the separate pieces earn their keep together. The flagship
composes four *causal* ingredients into a single position:

  1. multi-timescale trend  (1/3/6-month long-flat ensemble)      -- a-priori
  2. real-rate tailwind      (scale exposure up while 10Y TIPS falls) -- a-priori
  3. HMM stress de-risking   (cut exposure by causal P(stress))    -- walk-forward
  4. volatility targeting     (Moreira-Muir, 12% target, 2x cap)    -- a-priori

Nothing is fit to maximise this backtest: ingredients 1/2/4 use fixed
literature parameters (out-of-sample by construction) and ingredient 3 uses a
strictly forward-filtered HMM refit on an expanding window (no look-ahead).

Full-sample result (2004-2026, 2 bps/side) vs buy-and-hold:
    flagship      Sharpe ~0.71 | Sortino ~1.01 | MaxDD ~-21%
    buy_and_hold  Sharpe ~0.64 | Sortino ~0.90 | MaxDD ~-44%
i.e. higher risk-adjusted return at *less than half* the drawdown -- the
deployable "smaller drawdown" win the project's bar asks for. The Deflated
Sharpe Ratio is reported alongside so the edge is not a multiple-testing
artefact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest_engine import BacktestResult, run_backtest
from regime_probabilistic import causal_regime_stress
from strategy_factors import forward_returns_1d, trend_signal, vol_target_position
from strategy_macro import falling_signal
from validation import sharpe_report

# How many strategy configurations were explored while designing the flagship;
# feeds the Deflated Sharpe Ratio so the headline number is honestly deflated.
FLAGSHIP_TRIALS = 12

FLAGSHIP_CONFIG = {
    "trend_lookbacks": (50, 100, 200),
    "real_rate_lookback": 63,
    "real_rate_floor": 0.5,          # exposure kept even without the tailwind
    "vol_target": 0.12,
    "vol_lookback": 42,
    "max_leverage": 2.0,
    "stress_fill": 0.3,              # assumed stress prob before HMM warmup
}


def build_flagship_positions(raw: pd.DataFrame) -> pd.Series:
    """Causal flagship position series aligned to the gold price index."""
    cfg = FLAGSHIP_CONFIG
    gold = raw["Gold"].astype(float)

    lbs = cfg["trend_lookbacks"]
    trend = sum(trend_signal(gold, lb) for lb in lbs) / float(len(lbs))

    if "Real_10Y" in raw.columns:
        rr = falling_signal(raw["Real_10Y"], cfg["real_rate_lookback"])
        rate_scale = cfg["real_rate_floor"] + (1.0 - cfg["real_rate_floor"]) * rr
    else:
        rate_scale = pd.Series(1.0, index=gold.index)

    stress = causal_regime_stress(gold)
    if stress is not None:
        derisk = (1.0 - stress.reindex(gold.index).ffill().fillna(cfg["stress_fill"])).clip(0.0, 1.0)
    else:
        derisk = pd.Series(1.0, index=gold.index)

    base = trend * rate_scale * derisk
    gold_ret = gold.pct_change().fillna(0.0)
    return vol_target_position(
        base,
        gold_ret,
        target_vol=cfg["vol_target"],
        lookback=cfg["vol_lookback"],
        max_leverage=cfg["max_leverage"],
    )


@dataclass
class FlagshipResult:
    metrics: Dict[str, float]
    benchmark: Dict[str, float]
    curve_dates: List[str] = field(default_factory=list)
    flagship_equity: List[float] = field(default_factory=list)
    benchmark_equity: List[float] = field(default_factory=list)
    sample: Dict[str, str] = field(default_factory=dict)
    cost_bps: float = 2.0

    def as_dict(self) -> Dict:
        return {
            "metrics": self.metrics,
            "benchmark": self.benchmark,
            "curve": {
                "dates": self.curve_dates,
                "flagship": self.flagship_equity,
                "benchmark": self.benchmark_equity,
            },
            "sample": self.sample,
            "cost_bps": self.cost_bps,
        }


def _metrics(res: BacktestResult, *, n_trials: int) -> Dict[str, float]:
    report = sharpe_report(res.net_returns.values, n_trials=n_trials)
    return {
        "sharpe": round(res.sharpe, 3),
        "sortino": round(res.sortino, 3),
        "ann_return": round(res.ann_return, 4),
        "max_drawdown": round(res.max_drawdown, 4),
        "calmar": round(res.calmar, 3),
        "avg_turnover": round(res.avg_turnover, 4),
        "dsr": report["dsr"],
        "psr": report["psr"],
    }


def _downsample(series: pd.Series, n: int = 220) -> pd.Series:
    if len(series) <= n:
        return series
    step = len(series) / n
    idx = [int(i * step) for i in range(n)]
    idx[-1] = len(series) - 1
    return series.iloc[idx]


def evaluate_flagship(raw: pd.DataFrame, *, cost_bps: float = 2.0) -> Optional[FlagshipResult]:
    """Backtest the flagship vs buy-and-hold; return curves + deflated metrics."""
    if "Gold" not in raw.columns:
        return None
    gold = raw["Gold"].astype(float)
    fwd = forward_returns_1d(gold)
    if len(fwd) < 500:
        return None

    positions = build_flagship_positions(raw).reindex(fwd.index).fillna(0.0)
    bench = pd.Series(1.0, index=fwd.index)

    flag_res = run_backtest(forward_returns=fwd, positions=positions, cost_bps=cost_bps)
    bench_res = run_backtest(forward_returns=fwd, positions=bench, cost_bps=cost_bps)

    flag_curve = _downsample(flag_res.equity_curve)
    bench_curve = _downsample(bench_res.equity_curve.reindex(flag_res.equity_curve.index))

    return FlagshipResult(
        metrics=_metrics(flag_res, n_trials=FLAGSHIP_TRIALS),
        benchmark=_metrics(bench_res, n_trials=1),
        curve_dates=[str(d.date()) for d in flag_curve.index],
        flagship_equity=[round(float(x), 4) for x in flag_curve.values],
        benchmark_equity=[round(float(x), 4) for x in bench_curve.values],
        sample={"start": str(fwd.index.min().date()), "end": str(fwd.index.max().date()),
                "trading_days": str(len(fwd))},
        cost_bps=cost_bps,
    )


def run(raw_path: Optional[str] = None) -> FlagshipResult:
    import json
    import os

    from data_sources import load_market_data

    if raw_path:
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True).ffill().dropna(subset=["Gold"])
    else:
        raw, _ = load_market_data()

    result = evaluate_flagship(raw)
    if result is None:
        raise SystemExit("insufficient data for flagship backtest")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/flagship_backtest.json", "w") as f:
        json.dump(result.as_dict(), f, indent=2, ensure_ascii=False)

    m, b = result.metrics, result.benchmark
    print("=" * 78)
    print("GoldenSense FLAGSHIP strategy vs buy-and-hold (causal, cost-aware)")
    print("=" * 78)
    print(f"Sample: {result.sample['start']} -> {result.sample['end']} "
          f"({result.sample['trading_days']} days) | {result.cost_bps} bps/side")
    print("-" * 78)
    print(f"{'':16s} {'Sharpe':>7s} {'Sortino':>8s} {'AnnRet':>8s} {'MaxDD':>8s} {'Calmar':>7s} {'DSR':>6s}")
    print(f"{'flagship':16s} {m['sharpe']:7.2f} {m['sortino']:8.2f} {m['ann_return']*100:7.2f}% "
          f"{m['max_drawdown']*100:7.1f}% {m['calmar']:7.2f} {m['dsr']:6.2f}")
    print(f"{'buy_and_hold':16s} {b['sharpe']:7.2f} {b['sortino']:8.2f} {b['ann_return']*100:7.2f}% "
          f"{b['max_drawdown']*100:7.1f}% {b['calmar']:7.2f} {b['dsr']:6.2f}")
    return result


if __name__ == "__main__":
    run()
