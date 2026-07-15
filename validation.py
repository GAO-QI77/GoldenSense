"""Validation discipline upgrades: purged walk-forward and Deflated Sharpe.

Two failure modes this module guards against:

1. **Leakage through adjacent samples.** With overlapping/serially-correlated
   labels, a plain expanding split lets information bleed from train into
   test. ``purged_walk_forward_splits`` inserts an *embargo* gap between the
   end of each training slice and the start of its test block (Advances in
   Financial Machine Learning, ch. 7).

2. **Multiple-testing inflation.** Try enough strategy variants and one will
   sport a great Sharpe by luck. ``deflated_sharpe_ratio`` (Bailey &
   López de Prado, 2014) reports the probability that the observed Sharpe
   beats the expected maximum Sharpe of ``n_trials`` random tries, adjusting
   for non-normal returns. Report DSR next to every headline Sharpe.
"""
from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def purged_walk_forward_splits(
    n_samples: int,
    *,
    min_train: int,
    test_size: int,
    embargo: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits with an embargo gap before each test block.

    Train indices end ``embargo`` observations before the test block starts,
    so labels that overlap the boundary cannot leak forward.
    """
    if min_train <= 0 or test_size <= 0 or embargo < 0:
        raise ValueError("min_train/test_size must be positive, embargo >= 0")
    if min_train + embargo + 1 > n_samples:
        raise ValueError("not enough samples for even one split")

    test_start = min_train + embargo
    while test_start < n_samples:
        test_end = min(test_start + test_size, n_samples)
        train_end = test_start - embargo
        yield np.arange(0, train_end), np.arange(test_start, test_end)
        test_start = test_end


def probabilistic_sharpe_ratio(
    observed_sr: float,
    *,
    benchmark_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """P(true SR > benchmark_sr) given the observed SR and return moments.

    Sharpe ratios here are per-period (NOT annualized); ``kurtosis`` is the
    raw kurtosis (normal = 3).
    """
    if n_obs < 2:
        return 0.0
    denom = np.sqrt(
        max(1e-12, 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr ** 2)
    )
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] across ``n_trials`` zero-true-SR strategies (BLdP 2014)."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    std = np.sqrt(max(sr_variance, 1e-12))
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(std * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe_ratio(
    observed_sr: float,
    *,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    sr_variance: Optional[float] = None,
) -> float:
    """P(true SR > 0) after deflating for ``n_trials`` of strategy search.

    ``observed_sr`` is per-period. ``sr_variance`` is the variance of SR
    estimates across trials; when unknown we use the asymptotic estimator
    variance of a single SR, which is conservative for small trial counts.
    """
    if sr_variance is None:
        sr_variance = (
            1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr ** 2
        ) / max(n_obs - 1, 1)
    benchmark = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(
        observed_sr,
        benchmark_sr=benchmark,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurtosis,
    )


def annualized_to_daily_sr(annual_sr: float, periods_per_year: int = 252) -> float:
    return float(annual_sr / np.sqrt(periods_per_year))


def sharpe_report(
    net_returns: np.ndarray,
    *,
    n_trials: int,
    periods_per_year: int = 252,
) -> dict:
    """Convenience: annualized Sharpe + PSR + DSR from a net-return series."""
    net = np.asarray(net_returns, dtype=float)
    net = net[~np.isnan(net)]
    if len(net) < 30 or net.std(ddof=1) == 0:
        return {"sharpe_ann": 0.0, "psr": 0.0, "dsr": 0.0, "n_obs": int(len(net))}
    sr_daily = float(net.mean() / net.std(ddof=1))
    skew = float(stats.skew(net))
    kurt = float(stats.kurtosis(net, fisher=False))
    return {
        "sharpe_ann": round(sr_daily * np.sqrt(periods_per_year), 3),
        "psr": round(
            probabilistic_sharpe_ratio(
                sr_daily, benchmark_sr=0.0, n_obs=len(net), skew=skew, kurtosis=kurt
            ),
            4,
        ),
        "dsr": round(
            deflated_sharpe_ratio(
                sr_daily, n_trials=n_trials, n_obs=len(net), skew=skew, kurtosis=kurt
            ),
            4,
        ),
        "n_obs": int(len(net)),
        "skew": round(skew, 3),
        "kurtosis": round(kurt, 3),
    }
