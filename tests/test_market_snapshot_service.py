from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from data_loader import MarketDataLoader
from market_snapshot_service import MarketSnapshotConfig, build_market_snapshot, create_app


class _FakeMarketLoader:
    def fetch_data(self, period="6mo", interval="1d"):
        idx = pd.date_range(end=datetime.now(timezone.utc), periods=40, freq="D")
        return pd.DataFrame(
            {
                "Gold": np.linspace(2280.0, 2360.0, num=len(idx)),
                "USD_Index": np.linspace(105.0, 103.0, num=len(idx)),
                "VIX": np.linspace(16.0, 18.5, num=len(idx)),
                "10Y_Bond": np.linspace(4.4, 4.2, num=len(idx)),
                "2Y_Bond": np.linspace(4.8, 4.6, num=len(idx)),
                "S&P500": np.linspace(5000.0, 5100.0, num=len(idx)),
                "Crude_Oil": np.linspace(76.0, 81.0, num=len(idx)),
            },
            index=idx,
        )


class _FailingMarketLoader:
    def fetch_data(self, period="6mo", interval="1d"):
        raise RuntimeError("network_down")


class _JumpMarketLoader:
    provider_name = "test_jump"

    def fetch_data(self, period="6mo", interval="1d"):
        idx = pd.date_range(end=datetime.now(timezone.utc), periods=8, freq="D")
        return pd.DataFrame(
            {
                "Gold": [2300.0, 2310.0, 2368.0, 2375.0, 2320.0, 2332.0, 2388.0, 2394.0],
                "USD_Index": [104.0, 103.8, 102.1, 102.0, 104.4, 104.2, 102.9, 102.8],
                "VIX": [17.0, 17.2, 19.8, 18.9, 22.0, 20.5, 18.2, 18.0],
                "10Y_Bond": [4.4, 4.36, 4.18, 4.2, 4.47, 4.42, 4.22, 4.2],
                "2Y_Bond": [4.8, 4.78, 4.63, 4.64, 4.9, 4.86, 4.66, 4.65],
                "S&P500": [5000, 5010, 4960, 4980, 4890, 4920, 5015, 5030],
                "Crude_Oil": [78.0, 78.3, 81.2, 80.8, 76.5, 77.0, 79.6, 79.8],
            },
            index=idx,
        )


def test_build_market_snapshot_contract():
    snapshot = build_market_snapshot(_FakeMarketLoader().fetch_data())
    assert snapshot.asset == "XAUUSD"
    assert snapshot.latest_price > 0
    assert snapshot.feature_summary.technical_state in {"bullish", "bearish", "mixed"}
    assert snapshot.feature_summary.volatility_regime in {"calm", "elevated", "stress"}
    assert snapshot.feature_summary.ma5 is not None
    assert snapshot.feature_summary.ma20 is not None
    assert snapshot.feature_summary.ma60 is not None
    assert snapshot.feature_summary.rsi14 is not None
    assert snapshot.feature_summary.macd is not None
    assert snapshot.feature_summary.atr14_pct is not None
    assert any(item.symbol == "VIX" for item in snapshot.instruments)


def test_market_snapshot_endpoint_contract():
    app = create_app(market_loader=_FakeMarketLoader(), start_background_task=False)
    with TestClient(app) as client:
        refresh = client.post("/api/v1/market/snapshot/refresh")
        assert refresh.status_code == 200

        resp = client.get("/api/v1/market/snapshot/latest")
        assert resp.status_code == 200
        data = resp.json()
    assert data["asset"] == "XAUUSD"
    assert data["feature_summary"]["technical_state"] in {"bullish", "bearish", "mixed"}
    assert isinstance(data["instruments"], list)
    assert len(data["instruments"]) >= 3


def test_market_indicators_endpoint_groups_research_pillars():
    app = create_app(market_loader=_FakeMarketLoader(), start_background_task=False)
    with TestClient(app) as client:
        refresh = client.post("/api/v1/market/snapshot/refresh")
        assert refresh.status_code == 200

        resp = client.get("/api/v1/market/indicators/current")
        assert resp.status_code == 200
        data = resp.json()

    assert data["asset"] == "XAUUSD"
    assert {group["id"] for group in data["groups"]} == {
        "fundamental",
        "technical",
        "macro_policy",
        "flow_sentiment",
    }
    assert len(data["citations"]) >= 4
    technical = next(group for group in data["groups"] if group["id"] == "technical")
    technical_labels = {item["label"] for item in technical["indicators"]}
    assert {"MA5/20/60", "RSI", "MACD", "ATR/波动率", "关键支撑阻力"}.issubset(technical_labels)
    for group in data["groups"]:
        assert group["status"] in {"ok", "degraded", "unavailable"}
        assert group["freshness_seconds"] >= 0
        assert len(group["indicators"]) >= 4
        for indicator in group["indicators"]:
            assert indicator["source"]
            assert indicator["status"] in {"ok", "degraded", "unavailable"}
            assert indicator["freshness_seconds"] >= 0


def test_gold_history_endpoint_marks_two_percent_key_nodes_with_real_factor_context():
    app = create_app(market_loader=_JumpMarketLoader(), start_background_task=False)
    with TestClient(app) as client:
        resp = client.get("/api/v1/market/gold/history")
        assert resp.status_code == 200
        data = resp.json()

    assert data["asset"] == "XAUUSD"
    assert data["source"] == "test_jump"
    assert len(data["points"]) == 8
    assert len(data["key_nodes"]) >= 3
    for node in data["key_nodes"]:
        assert abs(node["change_pct"]) >= 0.02
        assert node["reason"]
        assert len(node["factors"]) >= 1


def test_market_snapshot_refresh_uses_synthetic_fallback_when_upstream_fails():
    app = create_app(market_loader=_FailingMarketLoader(), start_background_task=False)
    with TestClient(app) as client:
        resp = client.post("/api/v1/market/snapshot/refresh")
        assert resp.status_code == 200
        data = resp.json()
    assert data["latest_price"] > 0
    assert data["status"] == "degraded"
    assert data["degraded_reason"].startswith("synthetic_fallback:")
    assert any(item["source"] == "synthetic_fallback" for item in data["instruments"])


def test_market_readiness_fails_without_snapshot_when_fallback_disabled():
    app = create_app(
        market_loader=_FailingMarketLoader(),
        config=MarketSnapshotConfig(allow_synthetic_fallback=False),
        start_background_task=False,
    )
    with TestClient(app) as client:
        resp = client.get("/health/ready")
        data = resp.json()
    assert resp.status_code == 503
    assert data["status"] == "unavailable"
    assert "market_snapshot_unavailable" in data["errors"]


def test_yfinance_loader_uses_2y_yield_instead_of_13w_bill():
    loader = MarketDataLoader()
    assert loader.tickers["2Y_Bond"] == "2YY=F"
