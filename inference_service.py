import asyncio
from datetime import datetime, timezone
import os
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from stacking_model import DynamicEnsemble
from data_loader import MarketDataLoader, NewsDataLoader
from feature_engineer import FeatureEngineer


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_symbol: str = Field(min_length=1, max_length=20)
    horizon: str = Field(pattern=r"^T\+(1|7|30)$")
    current_timestamp: datetime


class FeatureImportanceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    importance: float


class AttentionLagItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lag: int
    weight: float


class ForecastResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction_prediction: int
    probability: float
    xgboost_direction_prediction: int
    xgboost_probability: float
    confidence_interval: Tuple[float, float]
    feature_importance_top_3: List[FeatureImportanceItem]
    attention_top_3_lags: List[AttentionLagItem]
    forecast_basis: str = "ensemble_model"
    supporting_reasons: List[str] = Field(default_factory=list, max_length=4)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_code: str
    message: str


def _prob_up_from_return(pred_return: float, scale: float) -> float:
    prob_up = 0.5 + float(np.tanh(pred_return * 10.0 * scale)) * 0.4
    return float(np.clip(prob_up, 0.0, 1.0))


def _direction_probability(direction: int, prob_up: float) -> float:
    if direction == 1:
        return prob_up
    return 1.0 - prob_up


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    w = weights / np.sum(weights)
    mean = float(np.dot(values, w))
    var = float(np.dot((values - mean) ** 2, w))
    return mean, float(np.sqrt(max(var, 0.0)))


def _feature_importance_top_3(
    model: DynamicEnsemble, X_tab_df: pd.DataFrame
) -> List[FeatureImportanceItem]:
    feature_names = list(X_tab_df.columns)
    xgb_model = getattr(model, "xgb", None)
    get_booster = getattr(xgb_model, "get_booster", None)
    if callable(get_booster):
        booster = xgb_model.get_booster()
        dm = xgb.DMatrix(X_tab_df.values, feature_names=feature_names)
        contribs = booster.predict(dm, pred_contribs=True)
        if isinstance(contribs, np.ndarray) and contribs.ndim == 2 and contribs.shape[0] == 1:
            contribs_1 = contribs[0]
            if contribs_1.shape[0] == len(feature_names) + 1:
                contribs_1 = contribs_1[:-1]
            order = np.argsort(np.abs(contribs_1))[::-1][:3]
            return [
                FeatureImportanceItem(feature=feature_names[i], importance=float(contribs_1[i]))
                for i in order
            ]

    row = X_tab_df.iloc[0].to_numpy(dtype=float)
    order = np.argsort(np.abs(row))[::-1][:3]
    return [FeatureImportanceItem(feature=feature_names[i], importance=float(row[i])) for i in order]


def _attention_top_3_lags(model: DynamicEnsemble, seq_len: int) -> List[AttentionLagItem]:
    transformer = getattr(model, "transformer", None)
    attn = getattr(transformer, "last_attention_weights", None)
    if attn is None:
        return []

    if isinstance(attn, torch.Tensor):
        attn_np = attn.detach().cpu().numpy()
    else:
        attn_np = np.asarray(attn)

    if attn_np.ndim != 4 or attn_np.shape[0] < 1:
        return []

    weights = attn_np[0, :, -1, :].mean(axis=0)
    if weights.shape[0] != seq_len:
        return []

    order = np.argsort(weights)[::-1][:3]
    items: List[AttentionLagItem] = []
    for i in order:
        lag = int(seq_len - 1 - int(i))
        items.append(AttentionLagItem(lag=lag, weight=float(weights[i])))
    return items


async def _to_thread_with_timeout(func, timeout_s: float):
    return await asyncio.wait_for(asyncio.to_thread(func), timeout=timeout_s)


def _series_return(series: pd.Series, periods: int) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return 0.0
    prev = float(clean.iloc[-periods - 1])
    last = float(clean.iloc[-1])
    if prev == 0:
        return 0.0
    return last / prev - 1.0


def _latest(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    return float(clean.iloc[-1])


def _synthetic_market_data(now: Optional[datetime] = None) -> pd.DataFrame:
    as_of = pd.Timestamp(now or datetime.now(timezone.utc)).tz_localize(None)
    idx = pd.date_range(end=as_of, periods=220, freq="D")
    steps = np.arange(len(idx), dtype=float)
    return pd.DataFrame(
        {
            "Gold": 2280.0 + steps * 2.6,
            "Silver": 28.0 + steps * 0.03,
            "USD_Index": 106.0 - steps * 0.03,
            "S&P500": 5000.0 + steps * 2.2,
            "VIX": 18.0 + np.sin(steps / 15.0) * 1.2,
            "Crude_Oil": 76.0 + steps * 0.05,
            "10Y_Bond": 4.5 - steps * 0.0015,
            "2Y_Bond": 4.9 - steps * 0.001,
        },
        index=idx,
    )


def _heuristic_signal_pack(market_df: pd.DataFrame, daily_signals: pd.DataFrame) -> Dict[str, float]:
    signal_pack = {
        "gold_return_5d": _series_return(market_df["Gold"], 5) if "Gold" in market_df else 0.0,
        "gold_return_20d": _series_return(market_df["Gold"], 20) if "Gold" in market_df else 0.0,
        "gold_return_60d": _series_return(market_df["Gold"], 60) if "Gold" in market_df else 0.0,
        "usd_return_5d": _series_return(market_df["USD_Index"], 5) if "USD_Index" in market_df else 0.0,
        "usd_return_20d": _series_return(market_df["USD_Index"], 20) if "USD_Index" in market_df else 0.0,
        "spx_return_20d": _series_return(market_df["S&P500"], 20) if "S&P500" in market_df else 0.0,
        "oil_return_20d": _series_return(market_df["Crude_Oil"], 20) if "Crude_Oil" in market_df else 0.0,
        "vix_level": (_latest(market_df["VIX"]) - 20.0) / 20.0 if "VIX" in market_df else 0.0,
        "yield_spread": (
            (_latest(market_df["10Y_Bond"]) - _latest(market_df["2Y_Bond"])) / 2.0
            if "10Y_Bond" in market_df and "2Y_Bond" in market_df
            else 0.0
        ),
        "news_total": 0.0,
        "news_risk": 0.0,
        "news_rates": 0.0,
    }
    if not daily_signals.empty:
        last = daily_signals.tail(1).iloc[0]
        signal_pack["news_total"] = float(last.get("total", 0.0))
        signal_pack["news_risk"] = float(last.get("risk", 0.0))
        signal_pack["news_rates"] = float(last.get("rates", 0.0))
    return signal_pack


def _supporting_reason(feature: str, contribution: float) -> str:
    direction = "支撑偏多" if contribution >= 0 else "压制偏多"
    mapping = {
        "gold_return_5d": "最近 5 个交易日的金价动量",
        "gold_return_20d": "最近 20 个交易日的金价趋势",
        "gold_return_60d": "最近 60 个交易日的中期趋势",
        "usd_return_5d": "美元短线变化",
        "usd_return_20d": "美元中期变化",
        "spx_return_20d": "风险偏好代理",
        "oil_return_20d": "原油趋势",
        "vix_level": "波动率水平",
        "yield_spread": "利率曲线斜率",
        "news_total": "最新新闻总情绪",
        "news_risk": "避险类新闻强度",
        "news_rates": "利率类新闻强度",
    }
    return f"{mapping.get(feature, feature)}正在{direction}。"


def _heuristic_forecast_response(
    *,
    horizon: str,
    market_df: pd.DataFrame,
    daily_signals: pd.DataFrame,
) -> ForecastResponse:
    signal_pack = _heuristic_signal_pack(market_df, daily_signals)
    configs = {
        "T+1": {
            "return_scale": 0.012,
            "prob_scale": 1.0,
            "ci_return_width": 0.008,
            "weights": {
                "gold_return_5d": 0.34,
                "usd_return_5d": -0.28,
                "news_total": 0.18,
                "news_rates": -0.12,
                "news_risk": 0.1,
                "vix_level": 0.06,
            },
        },
        "T+7": {
            "return_scale": 0.035,
            "prob_scale": 0.85,
            "ci_return_width": 0.02,
            "weights": {
                "gold_return_20d": 0.3,
                "usd_return_20d": -0.22,
                "yield_spread": -0.1,
                "news_total": 0.14,
                "news_risk": 0.12,
                "news_rates": -0.14,
                "spx_return_20d": -0.05,
            },
        },
        "T+30": {
            "return_scale": 0.07,
            "prob_scale": 0.7,
            "ci_return_width": 0.045,
            "weights": {
                "gold_return_60d": 0.24,
                "gold_return_20d": 0.14,
                "usd_return_20d": -0.18,
                "yield_spread": -0.14,
                "news_total": 0.08,
                "news_rates": -0.14,
                "spx_return_20d": -0.05,
                "oil_return_20d": 0.04,
                "vix_level": 0.03,
            },
        },
    }
    config = configs[horizon]
    contributions: List[Tuple[str, float]] = []
    raw_score = 0.0
    for feature, weight in config["weights"].items():
        value = float(signal_pack.get(feature, 0.0))
        contribution = weight * value
        contributions.append((feature, contribution))
        raw_score += contribution

    pred_return = float(np.tanh(raw_score * 6.0) * config["return_scale"])
    prob_up = _prob_up_from_return(pred_return, scale=config["prob_scale"])
    direction = 1 if pred_return >= 0.0 else -1
    probability = _direction_probability(direction, prob_up)

    xgb_pred_return = pred_return * 0.9
    xgb_prob_up = _prob_up_from_return(xgb_pred_return, scale=config["prob_scale"])
    xgb_direction = 1 if xgb_pred_return >= 0.0 else -1
    xgb_probability = _direction_probability(xgb_direction, xgb_prob_up)

    ci_low_r = pred_return - config["ci_return_width"]
    ci_high_r = pred_return + config["ci_return_width"]
    ci_low_p = _direction_probability(direction, _prob_up_from_return(ci_low_r, scale=config["prob_scale"]))
    ci_high_p = _direction_probability(direction, _prob_up_from_return(ci_high_r, scale=config["prob_scale"]))

    top_contributions = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:3]
    feature_importance = [
        FeatureImportanceItem(feature=name, importance=float(value))
        for name, value in top_contributions
    ]
    reasons = [_supporting_reason(name, value) for name, value in top_contributions]
    if horizon == "T+30":
        reasons.insert(0, "30 天视角当前采用代理预测，会更强调中期趋势、美元和利率环境。")

    return ForecastResponse(
        direction_prediction=direction,
        probability=float(probability),
        xgboost_direction_prediction=xgb_direction,
        xgboost_probability=float(xgb_probability),
        confidence_interval=(float(min(ci_low_p, ci_high_p)), float(max(ci_low_p, ci_high_p))),
        feature_importance_top_3=feature_importance,
        attention_top_3_lags=[],
        forecast_basis="heuristic_proxy",
        supporting_reasons=reasons[:4],
    )


def create_app(
    model_t1: Optional[DynamicEnsemble] = None,
    model_t7: Optional[DynamicEnsemble] = None,
    market_loader: Optional[MarketDataLoader] = None,
    news_loader: Optional[NewsDataLoader] = None,
    feature_engineer: Optional[FeatureEngineer] = None,
    model_checkpoints_dir_t1: str = "model_checkpoints",
    model_checkpoints_dir_t7: str = "model_checkpoints_t7",
) -> FastAPI:
    app = FastAPI()

    svc_models: Dict[str, DynamicEnsemble] = {
        "T+1": model_t1 or DynamicEnsemble(tabular_input_dim=15, seq_input_dim=4),
        "T+7": model_t7 or DynamicEnsemble(tabular_input_dim=15, seq_input_dim=4),
    }
    svc_market_loader = market_loader or MarketDataLoader()
    svc_news_loader = news_loader or NewsDataLoader()
    svc_feature_engineer = feature_engineer or FeatureEngineer()

    model_dirs: Dict[str, str] = {
        "T+1": model_checkpoints_dir_t1,
        "T+7": model_checkpoints_dir_t7,
    }
    model_loaded: Dict[str, bool] = {
        "T+1": model_t1 is not None,
        "T+7": model_t7 is not None,
    }
    model_load_lock = asyncio.Lock()
    model_loading_started: Dict[str, bool] = {
        "T+1": False,
        "T+7": False,
    }
    model_load_timeout_s = float(os.environ.get("INFERENCE_MODEL_LOAD_TIMEOUT_SECONDS", "6.0"))
    allow_synthetic_market_fallback = os.environ.get("INFERENCE_ALLOW_SYNTHETIC_FALLBACK", "1") != "0"

    async def _load_model_async(horizon: str) -> None:
        async with model_load_lock:
            if model_loaded[horizon]:
                model_loading_started[horizon] = False
                return
            model_dir = model_dirs[horizon]
            try:
                loaded = await _to_thread_with_timeout(
                    lambda: svc_models[horizon].load_model(model_dir), timeout_s=model_load_timeout_s
                )
                model_loaded[horizon] = bool(loaded)
            except Exception:
                model_loaded[horizon] = False
            finally:
                model_loading_started[horizon] = False

    async def _kick_model_load(horizon: str) -> None:
        if horizon not in model_loaded:
            return
        if model_loaded[horizon] or model_loading_started[horizon]:
            return
        model_loading_started[horizon] = True
        asyncio.create_task(_load_model_async(horizon))

    def _deps():
        return svc_models, svc_market_loader, svc_news_loader, svc_feature_engineer

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        if isinstance(exc.detail, dict) and "error_code" in exc.detail and "message" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"error_code": "http_error", "message": str(exc.detail)})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/v1/forecast", response_model=ForecastResponse)
    async def forecast(
        req: ForecastRequest,
        deps=Depends(_deps),
    ):
        svc_models_, svc_market_loader_, svc_news_loader_, svc_feature_engineer_ = deps
        horizon = req.horizon
        if horizon not in {"T+1", "T+7", "T+30"}:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error_code="unsupported_horizon",
                    message="Horizon must be T+1, T+7, or T+30.",
                ).model_dump(),
            )

        if horizon in {"T+1", "T+7"} and not model_loaded[horizon]:
            await _kick_model_load(horizon)

        if req.asset_symbol.upper() not in {"XAUUSD", "XAU/USD"}:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error_code="unsupported_asset",
                    message="Only XAUUSD is supported in this service.",
                ).model_dump(),
            )

        svc_model_ = svc_models_.get(horizon)

        try:
            market_df: pd.DataFrame = await _to_thread_with_timeout(
                lambda: svc_market_loader_.fetch_data(period="6mo", interval="1d"),
                timeout_s=20.0,
            )
        except Exception:
            if not allow_synthetic_market_fallback:
                raise HTTPException(
                    status_code=503,
                    detail=ErrorResponse(
                        error_code="market_data_unavailable",
                        message="Market data fetch failed.",
                    ).model_dump(),
                )
            market_df = _synthetic_market_data(req.current_timestamp)

        if market_df.empty or "Gold" not in market_df.columns:
            if not allow_synthetic_market_fallback:
                raise HTTPException(
                    status_code=503,
                    detail=ErrorResponse(
                        error_code="market_data_unavailable",
                        message="Insufficient market data for inference.",
                    ).model_dump(),
                )
            market_df = _synthetic_market_data(req.current_timestamp)

        as_of = pd.Timestamp(req.current_timestamp).tz_localize(None)
        market_df = market_df.loc[market_df.index <= as_of]
        if len(market_df) < 90:
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse(
                    error_code="insufficient_history",
                    message="Not enough history to build inference features.",
                ).model_dump(),
            )

        daily_signals = pd.DataFrame()
        try:
            news_items = await _to_thread_with_timeout(svc_news_loader_.fetch_news, timeout_s=10.0)
            scored_news = svc_news_loader_.analyze_causality(news_items)
            daily_signals = svc_news_loader_.get_daily_signals(scored_news)
        except Exception:
            daily_signals = pd.DataFrame()

        recent_data = market_df.tail(140)
        X_features = svc_feature_engineer_.prepare_inference_data(recent_data, daily_signals, n_rows=60)
        if X_features.empty or len(X_features) < 60:
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse(
                    error_code="feature_generation_failed",
                    message="Failed to generate inference features.",
                ).model_dump(),
            )

        seq_cols = ["Gold_ZScore", "Silver_ZScore", "Crude_Oil_ZScore", "USD_Index_ZScore"]
        if not all(c in X_features.columns for c in seq_cols):
            missing = [c for c in seq_cols if c not in X_features.columns]
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse(
                    error_code="insufficient_sequence_features",
                    message=f"Missing required sequence features: {missing}",
                ).model_dump(),
            )

        X_seq = X_features[seq_cols].tail(60).to_numpy(dtype=float).reshape(1, 60, 4)

        training_features: Sequence[str] = []
        if svc_model_ is not None and hasattr(svc_model_.xgb, "feature_names_in_"):
            training_features = list(svc_model_.xgb.feature_names_in_)
        if not training_features:
            training_features = svc_feature_engineer_.load_selected_features()
        if not training_features:
            training_features = list(X_features.columns)[:15]

        last_row = X_features.tail(1)
        X_tab_df = pd.DataFrame([{f: float(last_row[f].iloc[0]) if f in last_row.columns else 0.0 for f in training_features}])

        if horizon == "T+30" or svc_model_ is None or not model_loaded.get(horizon, False):
            return _heuristic_forecast_response(
                horizon=horizon,
                market_df=market_df,
                daily_signals=daily_signals,
            )

        try:
            l1_preds = svc_model_._get_l1_predictions(X_tab_df.values, X_seq)
        except Exception:
            return _heuristic_forecast_response(
                horizon=horizon,
                market_df=market_df,
                daily_signals=daily_signals,
            )

        weights = svc_model_.model_weights
        if weights is None:
            weights_arr = np.ones(l1_preds.shape[1], dtype=float)
        else:
            weights_arr = np.array(weights, dtype=float)
        base_vals = l1_preds[0].astype(float)
        pred_return, pred_std = _weighted_mean_std(base_vals, weights_arr)

        prob_up = _prob_up_from_return(pred_return, scale=1.0)
        direction = 1 if pred_return >= 0.0 else -1
        probability = _direction_probability(direction, prob_up)

        try:
            xgb_pred_return = float(svc_model_.xgb.predict(X_tab_df.values)[0])
        except Exception:
            xgb_pred_return = float(pred_return)

        xgb_prob_up = _prob_up_from_return(xgb_pred_return, scale=1.0)
        xgb_direction = 1 if xgb_pred_return >= 0.0 else -1
        xgb_probability = _direction_probability(xgb_direction, xgb_prob_up)

        z = 1.96
        ci_low_r = pred_return - z * pred_std
        ci_high_r = pred_return + z * pred_std
        ci_low_p = _direction_probability(direction, _prob_up_from_return(ci_low_r, scale=1.0))
        ci_high_p = _direction_probability(direction, _prob_up_from_return(ci_high_r, scale=1.0))
        ci_low = float(min(ci_low_p, ci_high_p))
        ci_high = float(max(ci_low_p, ci_high_p))

        fi_top3 = _feature_importance_top_3(svc_model_, X_tab_df)
        attn_top3 = _attention_top_3_lags(svc_model_, seq_len=60)

        return ForecastResponse(
            direction_prediction=direction,
            probability=float(probability),
            xgboost_direction_prediction=xgb_direction,
            xgboost_probability=float(xgb_probability),
            confidence_interval=(ci_low, ci_high),
            feature_importance_top_3=fi_top3,
            attention_top_3_lags=attn_top3,
            forecast_basis="ensemble_model",
            supporting_reasons=[f"{item.feature} 是当前模型最关注的特征。" for item in fi_top3],
        )

    return app


app = create_app()
