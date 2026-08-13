"""Local quantitative research context with TTL caching.

The gateway's per-request tools only see ~180 days of price history, which is
too short for the HMM regime model, the fair-value anchor, or Monte Carlo
cones. This module computes those from the repo-local long dataset
(``raw_market_data_extended.csv`` when present, the base CSV otherwise) and
caches the result for a TTL, so request paths stay fast and deterministic.

Everything degrades explicitly: each block is either present or listed in
``degraded`` with a reason. Nothing here ever fabricates data.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

DEFAULT_TTL_SECONDS = 6 * 3600
_FIT_WINDOW_OBS = 1500  # fit the HMM on the recent window; EM stays sub-second
# Daily EOD data is "fresh" if the last close is within a few days (a long
# weekend + a settlement lag). Beyond this the /quant page must say so.
DATA_STALE_AFTER_DAYS = 4


class LocalQuantContext:
    def __init__(self, *, ttl_seconds: float = DEFAULT_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._cache: Optional[Dict[str, Any]] = None
        self._computed_at: float = 0.0

    # ------------------------------------------------------------------ #
    def get_context(self, *, force_refresh: bool = False) -> Dict[str, Any]:
        with self._lock:
            fresh = (
                self._cache is not None
                and (time.time() - self._computed_at) < self._ttl
                and not force_refresh
            )
            if fresh:
                return self._cache
            self._cache = self._compute()
            self._computed_at = time.time()
            return self._cache

    def invalidate(self, *, min_age_seconds: float = 0.0) -> bool:
        """Drop the cache so the next request recomputes (event-driven
        refresh, e.g. after high-severity news). ``min_age_seconds`` is an
        anti-thrash guard: a cache younger than this is kept."""
        with self._lock:
            if self._cache is None:
                return False
            if (time.time() - self._computed_at) < min_age_seconds:
                return False
            self._cache = None
            self._computed_at = 0.0
            return True

    def latest_posterior(self) -> Optional[Dict[str, float]]:
        ctx = self.get_context()
        regime = ctx.get("regime_posterior")
        if regime:
            return regime.get("latest")
        return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute() -> Dict[str, Any]:
        import pandas as pd  # local imports keep gateway startup light

        context: Dict[str, Any] = {"degraded": {}, "computed_at": time.time()}

        try:
            from data_sources import EXTENDED_DATA_PATH, load_market_data

            raw, source = load_market_data()
            context["data_source"] = source
            asof = pd.Timestamp(raw.index.max())
            context["data_span"] = [
                str(raw.index.min().date()),
                str(asof.date()),
            ]
            context["data_rows"] = int(len(raw))

            # Honest freshness: this context is computed from a repo-local
            # dataset (built by data_sources.py), NOT live ticks. Surface how
            # old the last close is so the UI never implies it is real-time.
            now = pd.Timestamp(datetime.now(timezone.utc).date())
            age_days = int((now - asof.normalize()).days)
            context["data_asof"] = str(asof.date())
            context["data_age_days"] = age_days
            context["data_stale"] = age_days > DATA_STALE_AFTER_DAYS
            context["data_stale_after_days"] = DATA_STALE_AFTER_DAYS
            context["is_realtime"] = False
            # Per-series clock: a newly appended row may contain forward-filled
            # values from sources that have not published yet.  Surface each
            # series' last observed change instead of letting the frame's max
            # date masquerade as universal freshness.
            series_asof: Dict[str, str] = {}
            series_age_days: Dict[str, int] = {}
            for column in raw.columns:
                values = raw[column].dropna()
                if values.empty:
                    continue
                changed = values.ne(values.shift(1))
                last_change = pd.Timestamp(values.index[changed][-1])
                series_asof[str(column)] = str(last_change.date())
                series_age_days[str(column)] = int((now - last_change.normalize()).days)
            context["series_asof"] = series_asof
            context["series_age_days"] = series_age_days
            context["stale_series"] = sorted(
                name for name, age in series_age_days.items() if age > DATA_STALE_AFTER_DAYS
            )
            if source == "base":
                # The long extended dataset never made it into this deploy;
                # every quant block silently fell back to the short sample.
                context["degraded"]["extended_dataset"] = (
                    f"extended_dataset_missing:{EXTENDED_DATA_PATH}; "
                    "quant models fell back to the short base sample"
                )
        except Exception as exc:
            context["degraded"]["market_data"] = f"{type(exc).__name__}: {exc}"
            return context

        gold = raw["Gold"].astype(float).dropna()

        # --- HMM regime posterior (fitted on the recent window) ----------
        model = None
        try:
            from regime_probabilistic import GaussianHMM, STATE_LABELS, regime_features

            feats = regime_features(gold.iloc[-_FIT_WINDOW_OBS:])
            if len(feats) >= 400:
                model = GaussianHMM(n_states=3, max_iter=40).fit(feats.values)
                gamma = model.posterior(feats.values)
                labels = [STATE_LABELS[k] for k in range(3)]
                tail = gamma[-260:]
                context["regime_posterior"] = {
                    "latest": {
                        labels[k]: round(float(gamma[-1, k]), 4) for k in range(3)
                    },
                    "dates_tail": [str(d.date()) for d in feats.index[-260:]],
                    "series_tail": {
                        labels[k]: [round(float(x), 4) for x in tail[:, k]]
                        for k in range(3)
                    },
                }
            else:
                context["degraded"]["regime_posterior"] = "insufficient_history"
        except Exception as exc:
            context["degraded"]["regime_posterior"] = f"{type(exc).__name__}: {exc}"

        # --- Fair-value anchor -------------------------------------------
        deviation_pct = None
        try:
            from fair_value import fit_fair_value

            fv = fit_fair_value(raw)
            if fv is not None:
                context["fair_value"] = fv.as_dict()
                context["fair_value"]["deviation_series_tail"] = fv.deviation_series_tail
                deviation_pct = fv.deviation_pct
            else:
                context["degraded"]["fair_value"] = "macro_columns_or_history_missing"
        except Exception as exc:
            context["degraded"]["fair_value"] = f"{type(exc).__name__}: {exc}"

        # --- Volatility forecast & distribution bands ---------------------
        try:
            from vol_models import forecast_return_bands

            context["vol_bands"] = {
                "h1": forecast_return_bands(gold, horizon_days=1),
                "h5": forecast_return_bands(gold, horizon_days=5),
                "h21": forecast_return_bands(gold, horizon_days=21),
            }
        except Exception as exc:
            context["degraded"]["vol_bands"] = f"{type(exc).__name__}: {exc}"

        # --- Macro factor snapshot ----------------------------------------
        try:
            from strategy_macro import build_factor_signals

            signals, used = build_factor_signals(raw)
            latest = signals.iloc[-1]
            context["macro_factors"] = {
                "factors_used": used,
                "latest": {k: round(float(latest[k]), 4) for k in used},
                "composite": round(float(latest[used].mean()), 4),
            }
        except Exception as exc:
            context["degraded"]["macro_factors"] = f"{type(exc).__name__}: {exc}"

        # --- Monte Carlo scenario cone -------------------------------------
        try:
            from allocation import monte_carlo_cone

            cone = monte_carlo_cone(
                gold.iloc[-_FIT_WINDOW_OBS:],
                horizon_days=90,
                n_paths=1500,
                checkpoints=(30, 90),
                model=model,
            )
            if cone is not None:
                context["scenario_cone"] = cone
            else:
                context["degraded"]["scenario_cone"] = "insufficient_history"
        except Exception as exc:
            context["degraded"]["scenario_cone"] = f"{type(exc).__name__}: {exc}"

        # --- Flagship strategy backtest (causal, drawdown-aware) -----------
        # Walk-forward HMM refit makes this the heaviest block (~10s); the TTL
        # cache means users never pay it on the request path.
        try:
            from strategy_integrated import evaluate_flagship

            flagship = evaluate_flagship(raw)
            if flagship is not None:
                context["flagship"] = flagship.as_dict()
            else:
                context["degraded"]["flagship"] = "insufficient_history"
        except Exception as exc:
            context["degraded"]["flagship"] = f"{type(exc).__name__}: {exc}"

        # --- Cross-asset context --------------------------------------------
        try:
            from cross_asset import build_cross_asset_context

            cross = build_cross_asset_context(raw)
            if cross is not None:
                context["cross_asset"] = cross
            else:
                context["degraded"]["cross_asset"] = "insufficient_history"
        except Exception as exc:
            context["degraded"]["cross_asset"] = f"{type(exc).__name__}: {exc}"

        # --- Allocation advice per profile ---------------------------------
        try:
            from allocation import allocation_range

            posterior = None
            if context.get("regime_posterior"):
                posterior = context["regime_posterior"]["latest"]
            context["allocation"] = {
                profile: allocation_range(
                    profile,
                    regime_posterior=posterior,
                    valuation_deviation_pct=deviation_pct,
                ).as_dict()
                for profile in ("conservative", "balanced", "aggressive")
            }
        except Exception as exc:
            context["degraded"]["allocation"] = f"{type(exc).__name__}: {exc}"

        return context


# Process-wide singleton used by the gateway.
shared_context = LocalQuantContext()
