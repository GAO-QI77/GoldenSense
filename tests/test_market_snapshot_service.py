from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from market_snapshot_service import build_market_snapshot, create_app


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


def test_build_market_snapshot_contract():
    snapshot = build_market_snapshot(_FakeMarketLoader().fetch_data())
    assert snapshot.asset == "XAUUSD"
    assert snapshot.latest_price > 0
    assert snapshot.feature_summary.technical_state in {"bullish", "bearish", "mixed"}
    assert snapshot.feature_summary.volatility_regime in {"calm", "elevated", "stress"}
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


def test_market_snapshot_refresh_uses_synthetic_fallback_when_upstream_fails():
    app = create_app(market_loader=_FailingMarketLoader(), start_background_task=False)
    with TestClient(app) as client:
        resp = client.post("/api/v1/market/snapshot/refresh")
        assert resp.status_code == 200
        data = resp.json()
    assert data["latest_price"] > 0
    assert any(item["source"] == "synthetic_fallback" for item in data["instruments"])
