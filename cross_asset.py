"""Cross-asset context: where gold sits relative to the rest of the tape.

First honest step out of the single-asset ceiling: not new strategy sleeves
(each of those must earn its own walk-forward validation), but a deterministic
*context* block -- rolling correlations and trailing performance of gold
against the other assets already in the long dataset. It answers the C-end
user's real question ("我的其他资产和黄金是什么关系") without pretending to
have validated cross-asset signals.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Columns in the long dataset -> display labels.
PEER_ASSETS = {
    "Silver": "白银",
    "S&P500": "美股(标普500)",
    "USD_Index": "美元指数",
    "Crude_Oil": "原油",
    "10Y_Bond": "美债(10Y)",
}
CORR_WINDOW = 63          # ~one quarter of trading days
PERF_WINDOW = 252         # ~one trading year


def build_cross_asset_context(raw: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Rolling correlation (last value) + 1y trailing performance vs gold.

    Returns None when gold itself is missing; individual peers degrade
    per-asset (listed under ``missing``) rather than failing the block.
    """
    if "Gold" not in raw.columns:
        return None
    gold = raw["Gold"].astype(float).dropna()
    gold_ret = gold.pct_change().dropna()
    if len(gold_ret) < CORR_WINDOW + 5:
        return None

    peers: Dict[str, Any] = {}
    missing: Dict[str, str] = {}
    for column, label in PEER_ASSETS.items():
        if column not in raw.columns:
            missing[column] = "column_missing"
            continue
        series = raw[column].astype(float).dropna()
        ret = series.pct_change().dropna()
        aligned = pd.concat([gold_ret, ret], axis=1, join="inner").dropna()
        if len(aligned) < CORR_WINDOW + 5:
            missing[column] = "insufficient_overlap"
            continue
        corr = float(
            aligned.iloc[:, 0].rolling(CORR_WINDOW).corr(aligned.iloc[:, 1]).iloc[-1]
        )
        perf_window = series.iloc[-PERF_WINDOW:]
        perf_1y = float(perf_window.iloc[-1] / perf_window.iloc[0] - 1.0)
        peers[column] = {
            "label": label,
            "corr_63d": round(corr, 4) if np.isfinite(corr) else None,
            "perf_1y": round(perf_1y, 4),
        }

    gold_window = gold.iloc[-PERF_WINDOW:]
    return {
        "corr_window_days": CORR_WINDOW,
        "perf_window_days": PERF_WINDOW,
        "gold_perf_1y": round(float(gold_window.iloc[-1] / gold_window.iloc[0] - 1.0), 4),
        "peers": peers,
        "missing": missing,
        "note": (
            "背景对照口径：滚动相关与跟随表现仅描述历史统计关系，"
            "不构成跨资产配置信号；各资产策略需独立验证后方可上线。"
        ),
    }
