import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from feature_engineer import FeatureEngineer
from inference_service import create_app


class _FakeMarketDataLoader:
    def fetch_data(self, period="6mo", interval="1d"):
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        idx = pd.date_range(end=end, periods=180, freq="D")
        base = np.linspace(2000.0, 2400.0, num=len(idx))
        df = pd.DataFrame(
            {
                "Gold": base,
                "Silver": base / 80.0,
                "USD_Index": np.linspace(100.0, 105.0, num=len(idx)),
                "S&P500": np.linspace(4500.0, 4700.0, num=len(idx)),
                "VIX": np.linspace(15.0, 20.0, num=len(idx)),
                "Crude_Oil": np.linspace(70.0, 80.0, num=len(idx)),
                "10Y_Bond": np.linspace(4.0, 4.5, num=len(idx)),
                "2Y_Bond": np.linspace(5.0, 4.7, num=len(idx)),
            },
            index=idx,
        )
        return df


class _FakeNewsDataLoader:
    def __init__(self):
        self.fetch_calls = 0

    def fetch_news(self):
        self.fetch_calls += 1
        return []

    def analyze_causality(self, news_items):
        return []

    def get_daily_signals(self, scored_items):
        return pd.DataFrame()


class _FakeModel:
    def __init__(self):
        self.model_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

        class _X:
            feature_names_in_ = np.array(
                [
                    "Gold_ZScore",
                    "Silver_ZScore",
                    "Crude_Oil_ZScore",
                    "USD_Index_ZScore",
                    "VIX_ZScore",
                    "10Y_Bond_ZScore",
                    "2Y_Bond_ZScore",
                    "Gold_MA5_ZScore",
                    "Gold_MA20_ZScore",
                    "Gold_ATR_ZScore",
                    "Gold_Silver_Ratio_ZScore",
                    "Yield_Curve_Spread_ZScore",
                    "Gold_Return_1d_ZScore",
                    "Gold_Return_5d_ZScore",
                    "Gold_Momentum_ZScore",
                ]
            )
            
            def predict(self, X_tab):
                return np.array([0.005], dtype=float)

        self.xgb = _X()

    def load_model(self, path="model_checkpoints"):
        return True

    def _get_l1_predictions(self, X_tab, X_seq):
        return np.array([[0.01, 0.0, -0.005, 0.002]], dtype=float)


class _CountingMarketDataLoader(_FakeMarketDataLoader):
    def __init__(self):
        self.fetch_calls = 0

    def fetch_data(self, period="6mo", interval="1d"):
        self.fetch_calls += 1
        return super().fetch_data(period=period, interval=interval)


class _FailingLoadModel(_FakeModel):
    def __init__(self):
        super().__init__()
        self.load_calls = 0

    def load_model(self, path="model_checkpoints"):
        self.load_calls += 1
        raise TimeoutError("checkpoint load timeout")



def test_forecast_contract_ok():
    app = create_app(
        model_t1=_FakeModel(),
        model_t7=_FakeModel(),
        market_loader=_FakeMarketDataLoader(),
        news_loader=_FakeNewsDataLoader(),
        feature_engineer=FeatureEngineer(),
    )
    client = TestClient(app)

    payload = {
        "asset_symbol": "XAUUSD",
        "horizon": "T+1",
        "current_timestamp": datetime.now(UTC).isoformat(),
    }
    resp = client.post("/api/v1/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["direction_prediction"] in (1, -1)
    assert 0.0 <= float(data["probability"]) <= 1.0
    assert data["xgboost_direction_prediction"] in (1, -1)
    assert 0.0 <= float(data["xgboost_probability"]) <= 1.0
    assert isinstance(data["confidence_interval"], list)
    assert len(data["confidence_interval"]) == 2
    assert isinstance(data["feature_importance_top_3"], list)
    assert len(data["feature_importance_top_3"]) == 3
    for item in data["feature_importance_top_3"]:
        assert set(item.keys()) == {"feature", "importance"}
    assert "attention_top_3_lags" in data
    assert isinstance(data["attention_top_3_lags"], list)
    assert data["forecast_basis"] == "ensemble_model"
    assert data["model_status"] == "loaded"
    assert data["model_loaded"] is True
    assert data["model_checkpoint_path"] == "model_checkpoints"
    for item in data["attention_top_3_lags"]:
        assert set(item.keys()) == {"lag", "weight"}


def test_forecast_rejects_extra_fields():
    app = create_app(
        model_t1=_FakeModel(),
        model_t7=_FakeModel(),
        market_loader=_FakeMarketDataLoader(),
        news_loader=_FakeNewsDataLoader(),
        feature_engineer=FeatureEngineer(),
    )
    client = TestClient(app)

    payload = {
        "asset_symbol": "XAUUSD",
        "horizon": "T+1",
        "current_timestamp": datetime.now(UTC).isoformat(),
        "unexpected": 1,
    }
    resp = client.post("/api/v1/forecast", json=payload)
    assert resp.status_code == 422


def test_forecast_latency_under_200ms():
    app = create_app(
        model_t1=_FakeModel(),
        model_t7=_FakeModel(),
        market_loader=_FakeMarketDataLoader(),
        news_loader=_FakeNewsDataLoader(),
        feature_engineer=FeatureEngineer(),
    )
    client = TestClient(app)

    payload = {
        "asset_symbol": "XAUUSD",
        "horizon": "T+1",
        "current_timestamp": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
    }
    t0 = time.perf_counter()
    resp = client.post("/api/v1/forecast", json=payload)
    dt = time.perf_counter() - t0
    assert resp.status_code == 200
    assert dt < 0.2


def test_forecast_supports_t30_heuristic_proxy():
    app = create_app(
        model_t1=_FakeModel(),
        model_t7=_FakeModel(),
        market_loader=_FakeMarketDataLoader(),
        news_loader=_FakeNewsDataLoader(),
        feature_engineer=FeatureEngineer(),
    )
    client = TestClient(app)

    payload = {
        "asset_symbol": "XAUUSD",
        "horizon": "T+30",
        "current_timestamp": datetime.now(UTC).isoformat(),
    }
    resp = client.post("/api/v1/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["forecast_basis"] == "heuristic_proxy"
    assert data["model_status"] == "not_applicable"
    assert data["model_loaded"] is False
    assert len(data["supporting_reasons"]) >= 1


def test_missing_checkpoint_uses_live_heuristic_quant_engine_without_unavailable_status():
    app = create_app(
        market_loader=_FakeMarketDataLoader(),
        news_loader=_FakeNewsDataLoader(),
        feature_engineer=FeatureEngineer(),
        model_checkpoints_dir_t1="missing-t1",
        model_checkpoints_dir_t7="missing-t7",
    )
    client = TestClient(app)

    payload = {
        "asset_symbol": "XAUUSD",
        "horizon": "T+1",
        "current_timestamp": datetime.now(UTC).isoformat(),
    }
    resp = client.post("/api/v1/forecast", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["forecast_basis"] == "heuristic_proxy"
    assert data["model_status"] == "heuristic_proxy"
    assert data["model_loaded"] is False
    assert data["model_status"] != "unavailable"
    assert len(data["feature_importance_top_3"]) == 3
    assert len(data["supporting_reasons"]) >= 1


def test_inference_readiness_accepts_heuristic_proxy_when_checkpoints_are_missing():
    app = create_app(
        market_loader=_FakeMarketDataLoader(),
        news_loader=_FakeNewsDataLoader(),
        feature_engineer=FeatureEngineer(),
        model_checkpoints_dir_t1="missing-t1",
        model_checkpoints_dir_t7="missing-t7",
    )
    client = TestClient(app)

    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["forecast_mode"]["T+1"] == "heuristic_proxy"
    assert "T+1_checkpoint_missing" in data["warnings"]


def test_t7_default_checkpoint_uses_available_model_dir():
    app = create_app(
        market_loader=_FakeMarketDataLoader(),
        news_loader=_FakeNewsDataLoader(),
        feature_engineer=FeatureEngineer(),
    )
    assert app.state.model_dirs["T+7"] == "model_checkpoints"


def test_inference_readiness_reports_model_state():
    app = create_app(
        model_t1=_FakeModel(),
        model_t7=_FakeModel(),
        market_loader=_FakeMarketDataLoader(),
        news_loader=_FakeNewsDataLoader(),
        feature_engineer=FeatureEngineer(),
    )
    client = TestClient(app)
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_status"]["T+7"]["checkpoint_path"] == "model_checkpoints"


def test_batch_forecast_reuses_market_and_news_inputs_for_three_horizons():
    market_loader = _CountingMarketDataLoader()
    news_loader = _FakeNewsDataLoader()
    app = create_app(
        market_loader=market_loader,
        news_loader=news_loader,
        feature_engineer=FeatureEngineer(),
        model_checkpoints_dir_t1="missing-t1",
        model_checkpoints_dir_t7="missing-t7",
    )
    client = TestClient(app)

    resp = client.post(
        "/api/v1/forecast/batch",
        json={
            "asset_symbol": "XAUUSD",
            "horizons": ["T+1", "T+7", "T+30"],
            "current_timestamp": datetime.now(UTC).isoformat(),
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert set(data["forecasts"].keys()) == {"T+1", "T+7", "T+30"}
    assert all(item["forecast_basis"] == "heuristic_proxy" for item in data["forecasts"].values())
    assert data["forecasts"]["T+1"]["model_status"] == "heuristic_proxy"
    assert data["forecasts"]["T+7"]["model_status"] == "heuristic_proxy"
    assert data["forecasts"]["T+30"]["model_status"] == "not_applicable"
    assert market_loader.fetch_calls == 1
    assert news_loader.fetch_calls == 1


def test_model_load_cooldown_suppresses_repeated_checkpoint_attempts(monkeypatch):
    monkeypatch.setenv("INFERENCE_MODEL_LOAD_RETRY_COOLDOWN_SECONDS", "60")
    model = _FailingLoadModel()
    app = create_app(
        model_t1=model,
        market_loader=_FakeMarketDataLoader(),
        news_loader=_FakeNewsDataLoader(),
        feature_engineer=FeatureEngineer(),
    )
    app.state.model_load_retry_after["T+1"] = time.monotonic() + 60
    payload = {
        "asset_symbol": "XAUUSD",
        "horizon": "T+1",
        "current_timestamp": datetime.now(UTC).isoformat(),
    }

    with TestClient(app) as client:
        first = client.post("/api/v1/forecast", json=payload)
        time.sleep(0.1)
        second = client.post("/api/v1/forecast", json=payload)
        time.sleep(0.1)

    assert first.status_code == 200
    assert second.status_code == 200
    assert model.load_calls == 0
