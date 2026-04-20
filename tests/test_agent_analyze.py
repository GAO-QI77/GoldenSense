from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from agent_gateway import (
    AgentAnalysisService,
    AgentAnalyzeRequest,
    HistoricalEventsLookup,
    NarrativeOutput,
    RiskBanner,
    SummaryCard,
    create_app,
)
from service_contracts import InstrumentSnapshot, MarketFeatureSummary, MarketSnapshotResponse, NewsEventItem, RecentNewsResponse


PUBLIC_KEY = os.environ.get("AGENT_PUBLIC_API_KEYS", "dev-public-key").split(",")[0]
INTERNAL_KEY = os.environ.get("AGENT_INTERNAL_API_KEYS", "dev-internal-key").split(",")[0]


def _headers(scope: str = "public") -> dict[str, str]:
    return {"X-API-Key": INTERNAL_KEY if scope == "internal" else PUBLIC_KEY}


class _DraftNarrator:
    async def narrate(self, bundle, draft):
        return draft


class _ScenarioToolbox:
    def __init__(
        self,
        *,
        direction: int = 1,
        probability: float = 0.67,
        xgb_probability: float = 0.64,
        technical_state: str = "bullish",
        vix: float = 18.0,
        is_stale: bool = False,
        news_sentiment: float = 0.25,
        rag_t1: float = 0.012,
        rag_t7: float = 0.028,
        macro_signal: int = 1,
    ):
        self.direction = direction
        self.probability = probability
        self.xgb_probability = xgb_probability
        self.technical_state = technical_state
        self.vix = vix
        self.is_stale = is_stale
        self.news_sentiment = news_sentiment
        self.rag_t1 = rag_t1
        self.rag_t7 = rag_t7
        self.macro_signal = macro_signal

    async def get_market_snapshot(self):
        now = datetime.now(timezone.utc)
        return MarketSnapshotResponse(
            asset="XAUUSD",
            as_of=now,
            freshness_seconds=240 if self.is_stale else 12,
            stale_after_seconds=180,
            is_stale=self.is_stale,
            latest_price=2368.4,
            price_change_pct_1d=0.004,
            instruments=[
                InstrumentSnapshot(symbol="XAUUSD", label="黄金", price=2368.4, change_pct_1d=0.004, source="test", as_of=now),
                InstrumentSnapshot(symbol="VIX", label="波动率指数", price=self.vix, change_pct_1d=0.03, source="test", as_of=now),
                InstrumentSnapshot(symbol="DXY", label="美元指数", price=104.2, change_pct_1d=-0.002, source="test", as_of=now),
            ],
            feature_summary=MarketFeatureSummary(
                technical_state=self.technical_state,
                volatility_regime="stress" if self.vix >= 30 else "elevated" if self.vix >= 20 else "calm",
                yield_curve_spread=-0.4,
                gold_usd_divergence=0.01 if self.direction > 0 else -0.01,
                gold_momentum_5d=0.02 if self.direction > 0 else -0.02,
                stale_age_seconds=240 if self.is_stale else 12,
                is_stale=self.is_stale,
            ),
        )

    async def get_quant_forecast(self, horizon):
        return {
            "direction_prediction": self.direction,
            "probability": self.probability,
            "xgboost_direction_prediction": self.direction,
            "xgboost_probability": self.xgb_probability,
        }

    async def search_recent_news(self, query, limit=6):
        now = datetime.now(timezone.utc)
        return RecentNewsResponse(
            as_of=now,
            freshness_seconds=20,
            items=[
                NewsEventItem(
                    event_id="news-1",
                    published_at=now,
                    title=query,
                    summary="recent-news-summary",
                    source="test-wire",
                    normalized_event=query[:80],
                    sentiment_score=self.news_sentiment,
                    importance=0.8,
                    categories=["macro"],
                    url=None,
                )
            ],
        )

    async def retrieve_historical_events(self, text, top_k=3):
        from agent_gateway import RagEventItem

        return HistoricalEventsLookup(
            items=[
                RagEventItem(
                    headline=f"historical-{text[:12]}",
                    similarity=0.88,
                    gold_t1_return=self.rag_t1,
                    gold_t7_return=self.rag_t7,
                )
            ],
            status="ok",
            degraded_reason=None,
            source_freshness_seconds=None,
        )

    def get_macro_context(self, snapshot, news):
        return {
            "macro_signal": self.macro_signal,
            "avg_news_sentiment": self.news_sentiment,
            "dollar_message": "美元与黄金出现有利共振。",
            "rate_message": "利率曲线仍在提示对冲需求。",
            "news_message": "新闻主线与当前方向大体一致。",
        }

    def get_user_risk_profile(self, profile):
        labels = {
            "conservative": "保守型",
            "balanced": "平衡型",
            "aggressive": "进取型",
        }
        return {
            "profile": profile,
            "label": labels[profile],
            "preferred_action": "小仓试探" if profile == "conservative" else "分批布局",
            "max_action": "小仓试探" if profile == "conservative" else "分批布局",
            "description": "test-risk-profile",
        }


class _PartiallyFailingToolbox(_ScenarioToolbox):
    async def get_quant_forecast(self, horizon):
        raise RuntimeError("forecast_down")

    async def retrieve_historical_events(self, text, top_k=3):
        raise RuntimeError("memory_down")


def _make_client(toolbox: _ScenarioToolbox) -> TestClient:
    app = create_app(toolbox=toolbox, narrator=_DraftNarrator())
    return TestClient(app)


def test_agent_analyze_contract_ok():
    with _make_client(_ScenarioToolbox()) as client:
        payload = {
            "question": "CPI 超预期之后黄金怎么看？",
            "optional_news_text": "美国 CPI 超预期，市场重新评估降息路径。",
            "risk_profile": "conservative",
            "horizon": "24h",
            "locale": "zh-CN",
        }
        resp = client.post("/api/v1/agent/analyze", json=payload, headers=_headers())
        assert resp.status_code == 200
        data = resp.json()

    assert set(data.keys()) == {
        "analysis_id",
        "summary_card",
        "horizon_forecasts",
        "recent_news",
        "evidence_cards",
        "citations",
        "risk_banner",
        "degradation_flags",
        "follow_up_questions",
        "timing_ms",
    }
    assert isinstance(data["analysis_id"], str) and data["analysis_id"]
    assert data["summary_card"]["action"] in {"观望", "小仓试探", "分批布局", "降低暴露"}
    assert data["summary_card"]["stance"] in {"偏多", "偏空", "中性", "高风险观望"}
    assert data["summary_card"]["horizon"] == "24h"
    assert len(data["horizon_forecasts"]) == 3
    assert {item["horizon"] for item in data["horizon_forecasts"]} == {"24h", "7d", "30d"}
    assert len(data["recent_news"]) >= 1
    assert len(data["summary_card"]["reasons"]) >= 2
    assert len(data["summary_card"]["invalidators"]) >= 2
    assert len(data["evidence_cards"]) >= 1
    assert len(data["citations"]) >= 1
    assert data["risk_banner"]["level"] in {"low", "medium", "high"}
    assert all(isinstance(item, str) and item for item in data["follow_up_questions"])


def test_agent_analyze_rejects_extra_fields():
    with _make_client(_ScenarioToolbox()) as client:
        payload = {
            "question": "test",
            "risk_profile": "conservative",
            "horizon": "24h",
            "locale": "zh-CN",
            "unexpected": True,
        }
        resp = client.post("/api/v1/agent/analyze", json=payload, headers=_headers())
        assert resp.status_code == 422


def test_agent_analyze_supports_cors_preflight():
    with _make_client(_ScenarioToolbox()) as client:
        resp = client.options(
            "/api/v1/agent/analyze",
            headers={
                "Origin": "http://localhost:4173",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:4173"


def test_agent_analyze_rejects_empty_evidence(monkeypatch):
    def _empty_cards(self, bundle, citations):
        return []

    monkeypatch.setattr(AgentAnalysisService, "_build_evidence_cards", _empty_cards)
    with _make_client(_ScenarioToolbox()) as client:
        resp = client.post(
            "/api/v1/agent/analyze",
            json={
                "question": "test",
                "risk_profile": "conservative",
                "horizon": "24h",
                "locale": "zh-CN",
            },
            headers=_headers(),
        )
        assert resp.status_code == 503
        detail = resp.json()["detail"]
    assert detail["error_code"] == "insufficient_evidence"


def test_agent_endpoints_require_api_keys():
    with _make_client(_ScenarioToolbox()) as client:
        analyze = client.post(
            "/api/v1/agent/analyze",
            json={
                "question": "auth",
                "risk_profile": "conservative",
                "horizon": "24h",
                "locale": "zh-CN",
            },
        )
        assert analyze.status_code == 401

        feedback = client.post(
            "/api/v1/agent/feedback",
            json={"analysis_id": "missing", "rating": "helpful", "comment": None},
        )
        assert feedback.status_code == 401

        trace = client.get("/api/v1/agent/traces/missing")
        assert trace.status_code == 401

        trigger = client.post(
            "/api/v1/agent/trigger",
            json={"news_text": "auth", "manual_vix": 20.0},
        )
        assert trigger.status_code == 401


def test_internal_endpoints_reject_public_api_keys():
    with _make_client(_ScenarioToolbox()) as client:
        trace = client.get("/api/v1/agent/traces/missing", headers=_headers("public"))
        assert trace.status_code == 403

        trigger = client.post(
            "/api/v1/agent/trigger",
            json={"news_text": "auth", "manual_vix": 20.0},
            headers=_headers("public"),
        )
        assert trigger.status_code == 403


def test_legacy_trigger_respects_manual_vix_override():
    with _make_client(_ScenarioToolbox(direction=1, probability=0.72, news_sentiment=0.3, rag_t1=0.015, rag_t7=0.02)) as client:
        resp = client.post(
            "/api/v1/agent/trigger",
            json={
                "news_text": "地缘冲突升级推动避险需求。",
                "manual_vix": 35.0,
            },
            headers=_headers("internal"),
        )
        assert resp.status_code == 200
        data = resp.json()
    assert data["decision"]["action"] == "HOLD"
    assert data["risk_result"]["decision"] == "REJECTED"


def test_agent_feedback_records_trace():
    with _make_client(_ScenarioToolbox()) as client:
        analyze = client.post(
            "/api/v1/agent/analyze",
            json={
                "question": "反馈测试",
                "risk_profile": "conservative",
                "horizon": "24h",
                "locale": "zh-CN",
            },
            headers=_headers(),
        )
        assert analyze.status_code == 200
        analysis_id = analyze.json()["analysis_id"]

        trace_before = client.get(f"/api/v1/agent/traces/{analysis_id}", headers=_headers("internal"))
        assert trace_before.status_code == 200
        trace_data = trace_before.json()
        assert trace_data["analysis_id"] == analysis_id
        assert len(trace_data["tool_trace"]) >= 6

        feedback = client.post(
            "/api/v1/agent/feedback",
            json={
                "analysis_id": analysis_id,
                "rating": "helpful",
                "comment": "这条建议够清楚",
            },
            headers=_headers(),
        )
        assert feedback.status_code == 200
        data = feedback.json()
        trace_after = client.get(f"/api/v1/agent/traces/{analysis_id}", headers=_headers("internal"))
        assert trace_after.status_code == 200
        assert trace_after.json()["feedback_rating"] == "helpful"
    assert data == {"analysis_id": analysis_id, "status": "recorded"}


def test_agent_analyze_degrades_when_optional_tools_fail():
    with _make_client(_PartiallyFailingToolbox()) as client:
        resp = client.post(
            "/api/v1/agent/analyze",
            json={
                "question": "量化服务挂了怎么办",
                "risk_profile": "conservative",
                "horizon": "24h",
                "locale": "zh-CN",
            },
            headers=_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
    assert data["summary_card"]["action"] == "观望"
    assert data["risk_banner"]["title"] in {"降级模式", "高风险环境"}
    assert any(card["id"] == "ev-quant" for card in data["evidence_cards"])
    assert len(data["horizon_forecasts"]) == 3
    assert "quant_forecast_degraded" in data["degradation_flags"]


def test_optional_news_text_isolated_from_question_wording():
    with _make_client(_ScenarioToolbox()) as client:
        base_payload = {
            "optional_news_text": "美国 CPI 超预期，市场重新评估降息路径。",
            "risk_profile": "conservative",
            "horizon": "24h",
            "locale": "zh-CN",
        }
        resp_1 = client.post(
            "/api/v1/agent/analyze",
            json={**base_payload, "question": "我很紧张，这是不是要暴涨了？"},
            headers=_headers(),
        )
        resp_2 = client.post(
            "/api/v1/agent/analyze",
            json={**base_payload, "question": "我很害怕，这是不是要暴跌了？"},
            headers=_headers(),
        )
    assert resp_1.status_code == 200
    assert resp_2.status_code == 200
    data_1 = resp_1.json()
    data_2 = resp_2.json()
    assert data_1["summary_card"]["stance"] == data_2["summary_card"]["stance"]
    assert data_1["summary_card"]["action"] == data_2["summary_card"]["action"]
    assert data_1["recent_news"][0]["title"] == data_2["recent_news"][0]["title"] == base_payload["optional_news_text"]


def test_legacy_trigger_reuses_single_analysis_snapshot():
    with _make_client(_ScenarioToolbox()) as client:
        trigger = client.post(
            "/api/v1/agent/trigger",
            json={
                "news_text": "美国 CPI 超预期，市场重新评估降息路径。",
                "manual_vix": 19.0,
            },
            headers=_headers("internal"),
        )
        assert trigger.status_code == 200
        trace_id = trigger.headers["X-Trace-Analysis-Id"]
        trace = client.get(f"/api/v1/agent/traces/{trace_id}", headers=_headers("internal"))
        assert trace.status_code == 200

    trigger_data = trigger.json()
    trace_data = trace.json()
    bundle = trace_data["evidence_payload"]["bundle"]
    assert trigger_data["quant_probability"] == bundle["forecast"]["probability"]
    assert trigger_data["xgboost_probability"] == bundle["forecast"]["xgboost_probability"]
    assert trigger_data["rag_top_3_event_titles"] == [item["headline"] for item in bundle["rag_events"][:3]]


@pytest.mark.parametrize(
    ("name", "toolbox", "expected_action", "expected_stance"),
    [
        ("cpi_hot", _ScenarioToolbox(direction=1, probability=0.69, news_sentiment=0.32, rag_t1=0.014, rag_t7=0.03, macro_signal=1), "小仓试探", "偏多"),
        ("fomc_hawkish", _ScenarioToolbox(direction=-1, probability=0.68, xgb_probability=0.31, technical_state="bearish", news_sentiment=-0.28, rag_t1=-0.013, rag_t7=-0.026, macro_signal=-1), "降低暴露", "偏空"),
        ("fomc_dovish", _ScenarioToolbox(direction=1, probability=0.66, technical_state="bullish", news_sentiment=0.24, rag_t1=0.011, rag_t7=0.022, macro_signal=1), "小仓试探", "偏多"),
        ("geopolitics", _ScenarioToolbox(direction=1, probability=0.63, technical_state="bullish", news_sentiment=0.3, rag_t1=0.01, rag_t7=0.02, macro_signal=1), "小仓试探", "偏多"),
        ("dollar_surge", _ScenarioToolbox(direction=-1, probability=0.65, technical_state="bearish", news_sentiment=-0.24, rag_t1=-0.009, rag_t7=-0.019, macro_signal=-1), "降低暴露", "偏空"),
        ("vix_breaker", _ScenarioToolbox(direction=1, probability=0.7, technical_state="bullish", vix=34.0, news_sentiment=0.22, rag_t1=0.01, rag_t7=0.03, macro_signal=1), "观望", "高风险观望"),
        ("conflict", _ScenarioToolbox(direction=1, probability=0.67, technical_state="bullish", news_sentiment=-0.25, rag_t1=-0.015, rag_t7=-0.02, macro_signal=-1), "观望", "高风险观望"),
    ],
)
def test_agent_analyze_scenarios(name, toolbox, expected_action, expected_stance):
    with _make_client(toolbox) as client:
        resp = client.post(
            "/api/v1/agent/analyze",
            json={
                "question": name,
                "optional_news_text": name,
                "risk_profile": "conservative",
                "horizon": "24h",
                "locale": "zh-CN",
            },
            headers=_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
    assert data["summary_card"]["action"] == expected_action
    assert data["summary_card"]["stance"] == expected_stance
