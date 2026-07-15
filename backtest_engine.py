"""Cost-aware backtest engine for daily-rebalanced directional strategies.

Model-agnostic on purpose: it takes a position series and the realized 1-day
forward returns, and reports the metrics that actually decide whether a strategy
makes money after costs (Sharpe, Sortino, max drawdown, Calmar, turnover).

Convention
----------
- ``positions[t]``        : position held into date ``t``'s forward return,
                            expressed in units of the asset (e.g. -1..+1).
- ``forward_returns[t]``  : realized return earned from holding the asset over
                            the bar starting at ``t`` (i.e. close[t] -> close[t+1]).
- ``cost_bps``            : cost charged per unit of turnover, in basis points.
                            Turnover[t] = |positions[t] - positions[t-1]|, with the
                            first day measured against a flat (0) book.

There is no look-ahead: ``positions[t]`` must be decided using information known
at or before ``t``; this engine never shifts the position forward for you.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    gross_returns: pd.Series
    costs: pd.Series
    net_returns: pd.Series
    equity_curve: pd.Series
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    gross_sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    avg_turnover: float
    num_trades: int
    periods_per_year: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "total_return": self.total_return,
            "ann_return": self.ann_return,
            "ann_vol": self.ann_vol,
            "sharpe": self.sharpe,
            "gross_sharpe": self.gross_sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "hit_rate": self.hit_rate,
            "avg_turnover": self.avg_turnover,
            "num_trades": self.num_trades,
        }


def _annualized_sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    std = returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def _annualized_sortino(returns: np.ndarray, periods_per_year: int) -> float:
    if len(returns) < 2:
        return 0.0
    downside = np.minimum(returns, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev == 0 or np.isnan(downside_dev):
        return 0.0
    return float(returns.mean() / downside_dev * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def run_backtest(
    *,
    forward_returns: pd.Series,
    positions: pd.Series,
    cost_bps: float = 0.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    if len(forward_returns) != len(positions):
        raise ValueError(
            f"forward_returns ({len(forward_returns)}) and positions "
            f"({len(positions)}) must have the same length"
        )

    positions = positions.astype(float)
    forward_returns = forward_returns.astype(float)

    gross_returns = positions * forward_returns.values

    prev_positions = positions.shift(1).fillna(0.0)
    turnover = (positions - prev_positions).abs()
    costs = turnover * (cost_bps / 1e4)
    net_returns = gross_returns - costs.values

    equity_curve = (1.0 + net_returns).cumprod()

    net = net_returns.values
    n = len(net)
    total_return = float(equity_curve.iloc[-1] - 1.0) if n else 0.0
    if n and equity_curve.iloc[-1] > 0:
        ann_return = float(equity_curve.iloc[-1] ** (periods_per_year / n) - 1.0)
    else:
        ann_return = -1.0
    ann_vol = float(net.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else 0.0

    sharpe = _annualized_sharpe(net, periods_per_year)
    gross_sharpe = _annualized_sharpe(gross_returns.values, periods_per_year)
    sortino = _annualized_sortino(net, periods_per_year)
    max_dd = _max_drawdown(equity_curve)
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    hit_rate = float((net > 0).mean()) if n else 0.0
    avg_turnover = float(turnover.mean()) if n else 0.0
    num_trades = int((turnover.values > 1e-9).sum())

    return BacktestResult(
        gross_returns=gross_returns,
        costs=costs,
        net_returns=net_returns,
        equity_curve=equity_curve,
        total_return=total_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        gross_sharpe=gross_sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        hit_rate=hit_rate,
        avg_turnover=avg_turnover,
        num_trades=num_trades,
        periods_per_year=periods_per_year,
    )
