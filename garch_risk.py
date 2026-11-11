"""Offline Student-t GARCH forecasts using a rolling, point-in-time fit."""
from __future__ import annotations

import numpy as np
import pandas as pd

FIT_WINDOW = 756
REFIT_EVERY = 21


def walk_forward_garch(
    prices: pd.Series, *, horizons: tuple[int, ...] = (1, 5, 21), min_history: int = 260,
) -> pd.DataFrame:
    """Return forward sums of daily conditional variance in decimal units."""
    from arch import arch_model

    if not horizons or any(horizon < 1 for horizon in horizons) or not 100 <= min_history <= FIT_WINDOW:
        raise ValueError("GARCH requires positive horizons and 100 to 756 observations")
    if not isinstance(prices.index, pd.DatetimeIndex) or (
        prices.index.hasnans or not prices.index.is_unique or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("prices must have unique, increasing timestamps")
    prices = prices.astype(float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("prices must be finite and positive")
    returns = prices.pct_change(fill_method=None).iloc[1:] * 100.0
    values = returns.to_numpy()
    result = np.full((len(returns), len(horizons)), np.nan)
    parameters = None
    conditional_variance = np.nan

    for i in range(min_history - 1, len(returns)):
        if (i - min_history + 1) % REFIT_EVERY == 0:
            history = values[max(0, i - FIT_WINDOW + 1):i + 1]
            parameters = None
            if not np.any(history):
                continue
            try:
                fit = arch_model(
                    history, mean="Zero", vol="GARCH", p=1, q=1, dist="StudentsT", rescale=False,
                ).fit(disp="off", show_warning=False, options={"maxiter": 400})
            except (ValueError, np.linalg.LinAlgError):
                continue
            omega, alpha, beta, nu = np.asarray(fit.params, dtype=float)
            conditional_variance = float(fit.conditional_volatility[-1]) ** 2
            if (
                fit.convergence_flag == 0 and np.isfinite([omega, alpha, beta, nu, conditional_variance]).all()
                and omega > 0 and min(alpha, beta) >= 0 and alpha + beta <= 1 + 1e-8
                and nu > 2 and conditional_variance > 0
            ):
                parameters = (omega, alpha, beta)
        if parameters is None:
            continue
        omega, alpha, beta = parameters
        next_variance = omega + alpha * values[i] ** 2 + beta * conditional_variance
        conditional_variance = next_variance
        forward = np.empty(max(horizons))
        forward[0] = next_variance
        for step in range(1, len(forward)):
            forward[step] = omega + (alpha + beta) * forward[step - 1]
        total = np.cumsum(forward) / 10_000
        result[i] = [total[horizon - 1] for horizon in horizons]

    return pd.DataFrame(result, index=returns.index, columns=horizons)
