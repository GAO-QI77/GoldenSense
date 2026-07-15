"""Tests for the cost-aware backtest engine.

All expected values here are hand-computed so the test proves the engine's
arithmetic, not just that it runs. A backtest that is silently wrong produces a
fake Sharpe ratio, so these are deliberately concrete.
"""
import numpy as np
import pandas as pd
import pytest

from backtest_engine import run_backtest


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_gross_return_is_position_times_forward_return():
    positions = pd.Series([1.0, 0.0, 1.0], index=_idx(3))
    fwd = pd.Series([0.10, 0.20, 0.30], index=_idx(3))

    result = run_backtest(forward_returns=fwd, positions=positions, cost_bps=0.0)

    assert list(np.round(result.gross_returns.values, 10)) == [0.10, 0.0, 0.30]
    # No cost -> net equals gross.
    assert list(np.round(result.net_returns.values, 10)) == [0.10, 0.0, 0.30]


def test_turnover_cost_is_charged_on_position_change():
    # Enter from flat (turnover 1), flip to -1 (turnover 2), flip to +1 (turnover 2).
    positions = pd.Series([1.0, -1.0, 1.0], index=_idx(3))
    fwd = pd.Series([0.0, 0.0, 0.0], index=_idx(3))

    result = run_backtest(forward_returns=fwd, positions=positions, cost_bps=10.0)

    # 10 bps = 0.001 per unit turnover.
    expected_costs = [0.001 * 1, 0.001 * 2, 0.001 * 2]
    assert list(np.round(result.costs.values, 10)) == expected_costs
    assert list(np.round(result.net_returns.values, 10)) == [-c for c in expected_costs]


def test_equity_curve_compounds_net_returns():
    positions = pd.Series([1.0, 1.0, 1.0], index=_idx(3))
    fwd = pd.Series([0.50, -0.50, 0.0], index=_idx(3))

    result = run_backtest(forward_returns=fwd, positions=positions, cost_bps=0.0)

    # 1 -> 1.5 -> 0.75 -> 0.75
    assert list(np.round(result.equity_curve.values, 10)) == [1.5, 0.75, 0.75]
    assert result.total_return == pytest.approx(-0.25)


def test_max_drawdown_is_peak_to_trough():
    positions = pd.Series([1.0, 1.0, 1.0], index=_idx(3))
    fwd = pd.Series([0.50, -0.50, 0.0], index=_idx(3))

    result = run_backtest(forward_returns=fwd, positions=positions, cost_bps=0.0)

    # Peak 1.5, trough 0.75 -> drawdown -0.5.
    assert result.max_drawdown == pytest.approx(-0.5)


def test_perfect_foresight_has_positive_sharpe_and_full_hit_rate():
    fwd = pd.Series([0.01, -0.02, 0.03, -0.01], index=_idx(4))
    positions = pd.Series(np.sign(fwd.values), index=_idx(4))

    result = run_backtest(forward_returns=fwd, positions=positions, cost_bps=0.0)

    # Every bet is correct -> every gross return positive.
    assert (result.net_returns.values > 0).all()
    assert result.hit_rate == pytest.approx(1.0)
    assert result.sharpe > 0


def test_sharpe_matches_definition():
    fwd = pd.Series([0.01, -0.01, 0.02, -0.005], index=_idx(4))
    positions = pd.Series([1.0, 1.0, 1.0, 1.0], index=_idx(4))

    result = run_backtest(
        forward_returns=fwd, positions=positions, cost_bps=0.0, periods_per_year=252
    )

    net = fwd.values
    expected = net.mean() / net.std(ddof=1) * np.sqrt(252)
    assert result.sharpe == pytest.approx(expected)


def test_turnover_and_trade_count():
    positions = pd.Series([1.0, 1.0, 0.0], index=_idx(3))
    fwd = pd.Series([0.0, 0.0, 0.0], index=_idx(3))

    result = run_backtest(forward_returns=fwd, positions=positions, cost_bps=0.0)

    # turnover: enter 1, hold 0, exit 1.
    assert result.num_trades == 2
    assert result.avg_turnover == pytest.approx((1 + 0 + 1) / 3)


def test_misaligned_lengths_raise():
    positions = pd.Series([1.0, 1.0], index=_idx(2))
    fwd = pd.Series([0.0, 0.0, 0.0], index=_idx(3))

    with pytest.raises(ValueError):
        run_backtest(forward_returns=fwd, positions=positions, cost_bps=0.0)
