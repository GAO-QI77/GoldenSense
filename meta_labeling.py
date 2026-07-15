"""Meta-labeling: the defensible role for ML in the short-term layer.

Stage A showed ML cannot predict *direction*. Meta-labeling (López de Prado,
Advances in Financial ML ch. 3) gives ML a much easier job instead:

- The **primary signal** stays rule-based (multi-timescale trend).
- ``triple_barrier_labels`` marks, for each primary entry, whether the trade
  hit its profit-target before its stop-loss / time limit -- a binary
  "was this signal worth taking" label with volatility-scaled barriers.
- A **secondary classifier** (XGBoost) predicts that label from state
  features; its probability *filters/sizes* the primary signal. It never
  originates positions.

Everything trains inside purged walk-forward folds (validation.py) and is
judged by the cost-aware backtest engine -- same bar as every other sleeve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest_engine import run_backtest
from strategy_factors import forward_returns_1d, trend_signal
from validation import purged_walk_forward_splits, sharpe_report

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Triple-barrier labelling
# --------------------------------------------------------------------------- #
def triple_barrier_labels(
    prices: pd.Series,
    entries: pd.Series,
    *,
    vol: Optional[pd.Series] = None,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding_days: int = 10,
) -> pd.Series:
    """Label each entry date: 1 if profit target hit before stop/time, else 0.

    Barriers are volatility-scaled: PT = +pt_mult * vol_h, SL = -sl_mult *
    vol_h where vol_h is the trailing daily vol scaled to the holding window.
    The time barrier settles on the sign of the terminal return.
    """
    prices = prices.astype(float)
    if vol is None:
        vol = prices.pct_change().rolling(22).std()
    vol_h = vol * np.sqrt(max_holding_days)

    labels: Dict[pd.Timestamp, float] = {}
    idx = prices.index
    pos_map = {ts: i for i, ts in enumerate(idx)}

    for ts in entries[entries > 0].index:
        i = pos_map.get(ts)
        if i is None or i + 1 >= len(idx):
            continue
        v = vol_h.iloc[i]
        if not np.isfinite(v) or v <= 0:
            continue
        entry_price = prices.iloc[i]
        pt, sl = pt_mult * v, -sl_mult * v
        window = prices.iloc[i + 1 : i + 1 + max_holding_days]
        rets = window / entry_price - 1.0

        label = None
        for r in rets:
            if r >= pt:
                label = 1.0
                break
            if r <= sl:
                label = 0.0
                break
        if label is None and len(rets):
            label = 1.0 if rets.iloc[-1] > 0 else 0.0
        if label is not None:
            labels[ts] = label

    return pd.Series(labels, dtype=float).sort_index()


def state_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Causal features describing the state in which a signal fires."""
    gold = raw["Gold"].astype(float)
    rets = gold.pct_change()
    feats = pd.DataFrame(index=raw.index)
    feats["vol_22d"] = rets.rolling(22).std() * np.sqrt(TRADING_DAYS)
    feats["ret_21d"] = gold.pct_change(21)
    feats["ret_63d"] = gold.pct_change(63)
    feats["trend_50"] = trend_signal(gold, 50)
    feats["trend_200"] = trend_signal(gold, 200)
    feats["dist_ma200"] = gold / gold.rolling(200).mean() - 1.0
    if "Real_10Y" in raw.columns:
        feats["real_rate_chg_63d"] = raw["Real_10Y"].astype(float).diff(63)
    if "USD_Index" in raw.columns:
        feats["usd_chg_63d"] = raw["USD_Index"].astype(float).pct_change(63)
    if "VIX" in raw.columns:
        feats["vix"] = raw["VIX"].astype(float)
    return feats


# --------------------------------------------------------------------------- #
# Walk-forward meta-labeler
# --------------------------------------------------------------------------- #
@dataclass
class MetaLabelReport:
    n_signals: int
    n_labeled: int
    oos_prob_auc: Optional[float]
    baseline: Dict
    filtered: Dict

    def as_dict(self) -> Dict:
        return {
            "n_signals": self.n_signals,
            "n_labeled": self.n_labeled,
            "oos_prob_auc": self.oos_prob_auc,
            "baseline": self.baseline,
            "filtered": self.filtered,
        }


def walk_forward_meta_probabilities(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    min_train: int = 200,
    test_size: int = 60,
    embargo: int = 10,
) -> pd.Series:
    """Out-of-sample P(signal pays off) for each labeled signal date."""
    from xgboost import XGBClassifier

    X = features.loc[labels.index].dropna()
    y = labels.loc[X.index]
    n = len(X)
    probs = pd.Series(index=X.index, dtype=float)
    if n < min_train + embargo + 1:
        return probs.dropna()

    for train_idx, test_idx in purged_walk_forward_splits(
        n, min_train=min_train, test_size=test_size, embargo=embargo
    ):
        y_train = y.iloc[train_idx]
        if y_train.nunique() < 2:
            continue
        model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
        model.fit(X.iloc[train_idx].values, y_train.values)
        probs.iloc[test_idx] = model.predict_proba(X.iloc[test_idx].values)[:, 1]
    return probs.dropna()


def meta_filtered_positions(
    base_positions: pd.Series,
    probabilities: pd.Series,
    *,
    threshold: float = 0.5,
) -> pd.Series:
    """Zero out (or keep) the primary position by meta-probability.

    Dates without an OOS probability keep the primary position -- the filter
    can only act where it has an honest out-of-sample opinion.
    """
    filtered = base_positions.copy().astype(float)
    aligned = probabilities.reindex(base_positions.index)
    kill = aligned.notna() & (aligned < threshold)
    filtered[kill] = 0.0
    return filtered


def run(raw_path: Optional[str] = None) -> MetaLabelReport:
    import json
    import os

    from data_sources import load_market_data

    if raw_path:
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True).ffill().dropna()
    else:
        raw, _ = load_market_data()

    gold = raw["Gold"].astype(float)
    fwd = forward_returns_1d(gold)

    # Primary signal: fresh 50d trend crossings (entry events).
    trend = trend_signal(gold, 50)
    entries = ((trend == 1.0) & (trend.shift(1) == 0.0)).astype(float)

    labels = triple_barrier_labels(gold, entries)
    feats = state_features(raw)
    # Signal-event samples are scarce (a few hundred over 20 years), so the
    # walk-forward folds are sized in events, not days.
    probs = walk_forward_meta_probabilities(
        feats, labels, min_train=80, test_size=25, embargo=5
    )

    auc = None
    joined = pd.concat([probs.rename("p"), labels.rename("y")], axis=1).dropna()
    if len(joined) >= 30 and joined["y"].nunique() == 2:
        from sklearn.metrics import roc_auc_score

        auc = round(float(roc_auc_score(joined["y"], joined["p"])), 4)

    base_positions = trend.reindex(fwd.index).fillna(0.0)
    filtered = meta_filtered_positions(base_positions, probs).reindex(fwd.index).fillna(0.0)

    base_res = run_backtest(forward_returns=fwd, positions=base_positions, cost_bps=2.0)
    filt_res = run_backtest(forward_returns=fwd, positions=filtered, cost_bps=2.0)

    report = MetaLabelReport(
        n_signals=int(entries.sum()),
        n_labeled=int(len(labels)),
        oos_prob_auc=auc,
        baseline=sharpe_report(base_res.net_returns.values, n_trials=1),
        filtered=sharpe_report(filt_res.net_returns.values, n_trials=2),
    )

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/meta_labeling_report.json", "w") as f:
        json.dump(report.as_dict(), f, indent=2, ensure_ascii=False)

    print("Meta-labeling report:")
    print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
