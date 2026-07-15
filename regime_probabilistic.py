"""Probabilistic market-regime model (Gaussian HMM, EM fitted, numpy only).

Replaces the hard 0.34/0.66 trend-score thresholds in ``regime_strategy`` with
a soft, probability-weighted view: the model emits per-state posterior
probabilities and downstream exposure is the *probability-weighted blend* of
per-state exposures. This removes threshold flip-flopping near regime borders
and gives the narrator an honest sentence like "72% probability we are in the
high-volatility state".

Implementation notes
--------------------
- 2..4 hidden states over [daily return, log realized vol] features.
- Diagonal Gaussian emissions, forward-backward in log space (stable).
- Deterministic quantile-based initialization -> reproducible fits.
- States are relabelled by ascending volatility so state 0 is always "calm".
- No external dependency (hmmlearn not required).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

_EPS = 1e-300


def regime_features(prices: pd.Series, vol_window: int = 10) -> pd.DataFrame:
    """Causal [return, log trailing vol] feature frame for the HMM."""
    prices = prices.astype(float).dropna()
    rets = prices.pct_change()
    vol = rets.rolling(vol_window).std()
    frame = pd.DataFrame({"ret": rets, "log_vol": np.log(vol)})
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def _log_gaussian(X: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Log density of each row of X under a diagonal Gaussian."""
    var = np.maximum(var, 1e-10)
    diff2 = (X - mean) ** 2 / var
    return -0.5 * (np.log(2 * np.pi * var).sum() + diff2.sum(axis=1))


@dataclass
class HMMFit:
    means: np.ndarray          # (K, D)
    variances: np.ndarray      # (K, D)
    transition: np.ndarray     # (K, K)
    initial: np.ndarray        # (K,)
    log_likelihood: float
    n_iter: int


class GaussianHMM:
    def __init__(self, n_states: int = 3, max_iter: int = 60, tol: float = 1e-4):
        if not 2 <= n_states <= 4:
            raise ValueError("n_states must be in [2, 4]")
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.fit_: Optional[HMMFit] = None

    # ------------------------------------------------------------------ #
    def _init_params(self, X: np.ndarray):
        """Deterministic init: split observations by volatility quantile."""
        K, D = self.n_states, X.shape[1]
        vol_col = X[:, -1]
        edges = np.quantile(vol_col, np.linspace(0, 1, K + 1))
        means = np.zeros((K, D))
        variances = np.ones((K, D))
        for k in range(K):
            lo, hi = edges[k], edges[k + 1]
            mask = (vol_col >= lo) & (vol_col <= hi)
            if mask.sum() < 2:
                mask = np.ones(len(X), dtype=bool)
            means[k] = X[mask].mean(axis=0)
            variances[k] = np.maximum(X[mask].var(axis=0), 1e-8)
        transition = np.full((K, K), 0.05 / max(K - 1, 1))
        np.fill_diagonal(transition, 0.95)
        initial = np.full(K, 1.0 / K)
        return means, variances, transition, initial

    def _forward_backward(self, log_b: np.ndarray, transition: np.ndarray, initial: np.ndarray):
        T, K = log_b.shape
        log_A = np.log(np.maximum(transition, _EPS))
        log_pi = np.log(np.maximum(initial, _EPS))

        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_b[0]
        for t in range(1, T):
            prev = log_alpha[t - 1][:, None] + log_A
            log_alpha[t] = log_b[t] + _logsumexp_rows(prev.T)

        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            nxt = log_A + (log_b[t + 1] + log_beta[t + 1])[None, :]
            log_beta[t] = _logsumexp_rows(nxt)

        log_likelihood = float(_logsumexp(log_alpha[-1]))
        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)

        # Pairwise state expectations for the transition update (vectorized
        # over time: (T-1, K, K) tensor is tiny for daily data).
        if T > 1:
            m = (
                log_alpha[:-1, :, None]
                + log_A[None, :, :]
                + (log_b[1:] + log_beta[1:])[:, None, :]
                - log_likelihood
            )
            xi_sum = np.exp(m).sum(axis=0)
        else:
            xi_sum = np.zeros((K, K))
        return gamma, xi_sum, log_likelihood

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < 50:
            raise ValueError("X must be (T>=50, D)")
        means, variances, transition, initial = self._init_params(X)

        prev_ll = -np.inf
        n_iter = 0
        # Degenerate EM steps (a state's weight collapsing on short/synthetic
        # data) can transiently over/underflow; the np.maximum guards keep the
        # result valid, so suppress the benign numpy warnings for clean logs.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for n_iter in range(1, self.max_iter + 1):
                log_b = np.column_stack(
                    [_log_gaussian(X, means[k], variances[k]) for k in range(self.n_states)]
                )
                gamma, xi_sum, ll = self._forward_backward(log_b, transition, initial)

                weights = gamma.sum(axis=0)  # (K,)
                means = (gamma.T @ X) / np.maximum(weights[:, None], 1e-10)
                for k in range(self.n_states):
                    diff2 = (X - means[k]) ** 2
                    variances[k] = np.maximum(
                        (gamma[:, k][:, None] * diff2).sum(axis=0) / max(weights[k], 1e-10),
                        1e-8,
                    )
                transition = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-10)
                initial = gamma[0] / max(gamma[0].sum(), 1e-10)

                if abs(ll - prev_ll) < self.tol * max(abs(prev_ll), 1.0):
                    prev_ll = ll
                    break
                prev_ll = ll

        # Relabel states by ascending volatility (last feature dimension).
        order = np.argsort(means[:, -1])
        self.fit_ = HMMFit(
            means=means[order],
            variances=variances[order],
            transition=transition[np.ix_(order, order)],
            initial=initial[order],
            log_likelihood=prev_ll,
            n_iter=n_iter,
        )
        return self

    def posterior(self, X: np.ndarray) -> np.ndarray:
        """Smoothed per-date state probabilities, shape (T, K).

        Uses the backward pass, so a value at t depends on future data -- fine
        for describing history, but NOT causal. Use ``filter_posterior`` for
        anything that feeds a backtest position.
        """
        f = self.fit_
        if f is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=float)
        log_b = np.column_stack(
            [_log_gaussian(X, f.means[k], f.variances[k]) for k in range(self.n_states)]
        )
        gamma, _, _ = self._forward_backward(log_b, f.transition, f.initial)
        return gamma

    def filter_posterior(self, X: np.ndarray) -> np.ndarray:
        """Causal forward-filtered probabilities P(state_t | X_{1..t}), (T, K).

        Only the forward pass -- the value at t uses data up to and including t,
        never the future. This is the honest input for a position series.
        """
        f = self.fit_
        if f is None:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=float)
        T = len(X)
        K = self.n_states
        log_A = np.log(np.maximum(f.transition, _EPS))
        log_pi = np.log(np.maximum(f.initial, _EPS))
        log_b = np.column_stack(
            [_log_gaussian(X, f.means[k], f.variances[k]) for k in range(K)]
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            log_alpha = np.zeros((T, K))
            log_alpha[0] = log_pi + log_b[0]
            for t in range(1, T):
                prev = log_alpha[t - 1][:, None] + log_A
                log_alpha[t] = log_b[t] + _logsumexp_rows(prev.T)
            # Normalize each row to a proper filtered distribution.
            filtered = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
        return filtered / filtered.sum(axis=1, keepdims=True)


def _logsumexp(v: np.ndarray) -> float:
    m = np.max(v)
    return float(m + np.log(np.sum(np.exp(v - m))))


def _logsumexp_rows(m: np.ndarray) -> np.ndarray:
    mx = m.max(axis=1, keepdims=True)
    return (mx + np.log(np.exp(m - mx).sum(axis=1, keepdims=True))).ravel()


# --------------------------------------------------------------------------- #
# Gold-specific convenience layer
# --------------------------------------------------------------------------- #
STATE_LABELS = {0: "calm", 1: "elevated", 2: "stress"}


def fit_gold_regime_posterior(
    prices: pd.Series,
    *,
    n_states: int = 3,
    min_obs: int = 400,
) -> Optional[pd.DataFrame]:
    """Fit the HMM on gold history; returns per-date posterior DataFrame.

    Columns are the human labels for each state ("calm"/"elevated"/"stress"
    for 3 states). Returns None when history is too short -- callers must fall
    back to the rule-based regime logic.
    """
    feats = regime_features(prices)
    if len(feats) < min_obs:
        return None
    model = GaussianHMM(n_states=n_states).fit(feats.values)
    gamma = model.posterior(feats.values)
    labels = [STATE_LABELS.get(k, f"state_{k}") for k in range(n_states)]
    return pd.DataFrame(gamma, index=feats.index, columns=labels)


def causal_regime_stress(
    prices: pd.Series,
    *,
    n_states: int = 3,
    min_train: int = 756,
    refit_every: int = 126,
) -> Optional[pd.Series]:
    """Causal P(stress) series for backtests: no look-ahead anywhere.

    Walk forward: refit the HMM on data up to time t (expanding window, every
    ``refit_every`` days), then take the *filtered* stress probability at t.
    Both the parameter estimation and the inference use only past data, so the
    resulting series can drive a position without leakage.

    Returns None when history is shorter than ``min_train``.
    """
    feats = regime_features(prices)
    if len(feats) < min_train + refit_every:
        return None

    values = feats.values
    idx = feats.index
    stress_col = n_states - 1  # states relabelled by ascending vol -> last = stress
    out = pd.Series(index=idx, dtype=float)

    start = min_train
    model = None
    while start < len(feats):
        end = min(start + refit_every, len(feats))
        # Refit on everything strictly before this block.
        try:
            model = GaussianHMM(n_states=n_states, max_iter=40).fit(values[:start])
        except Exception:
            if model is None:
                start = end
                continue
        # Filter over history+block, read only this block's filtered stress prob.
        filt = model.filter_posterior(values[:end])
        out.iloc[start:end] = filt[start:end, stress_col]
        start = end

    return out.dropna()


def blended_exposure(
    posterior_last: Dict[str, float],
    exposure_by_state: Dict[str, float],
) -> float:
    """Probability-weighted exposure blend: sum_k p_k * exposure_k."""
    total = 0.0
    for state, prob in posterior_last.items():
        total += float(prob) * float(exposure_by_state.get(state, 0.0))
    return float(np.clip(total, 0.0, 1.5))
