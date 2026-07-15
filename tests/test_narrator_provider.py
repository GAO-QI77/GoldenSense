from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from agent_gateway import (
    AgentGatewayConfig,
    AnalysisBundle,
    NarrativeOutput,
    OpenAINarrator,
    RiskBanner,
    SummaryCard,
)
from service_contracts import (
    InstrumentSnapshot,
    MarketFeatureSummary,
    MarketSnapshotResponse,
    RecentNewsResponse,
)


def _config() -> AgentGatewayConfig:
    return AgentGatewayConfig(
        forecast_url="http://localhost:8010/api/v1/forecast",
        memory_url="http://localhost:8012/api/v1/memory/search",
        market_snapshot_url="http://localhost:8014/api/v1/market/snapshot/latest",
        market_indicators_url="http://localhost:8014/api/v1/market/indicators/current",
        market_history_url="http://localhost:8014/api/v1/market/gold/history",
        recent_news_url="http://localhost:8016/api/v1/news/recent",
        default_model="deepseek-v4-flash",
        complex_model="deepseek-v4-pro",
        vix_circuit_breaker_threshold=30.0,
        stale_after_seconds=180,
        news_stale_after_seconds=300,
    )


def _draft(*, action: str = "观望") -> NarrativeOutput:
    return NarrativeOutput(
        summary_card=SummaryCard(
            stance="中性",
            horizon="24h",
            confidence_band="低",
            action=action,
            reasons=["规则原因一", "规则原因二"],
            invalidators=["失效条件一", "失效条件二"],
            disclaimer="规则回退",
        ),
        risk_banner=RiskBanner(level="medium", title="规则风险", message="规则风险说明"),
        follow_up_questions=["规则追问"],
    )


def _bundle(*, is_high_risk: bool = False) -> AnalysisBundle:
    now = datetime.now(timezone.utc)
    snapshot = MarketSnapshotResponse(
        asset="XAUUSD",
        as_of=now,
        freshness_seconds=12,
        stale_after_seconds=180,
        is_stale=False,
        latest_price=2368.4,
        price_change_pct_1d=0.004,
        instruments=[
            InstrumentSnapshot(
                symbol="XAUUSD",
                label="黄金",
                price=2368.4,
                change_pct_1d=0.004,
                source="test",
                as_of=now,
            )
        ],
        feature_summary=MarketFeatureSummary(
            technical_state="bullish",
            volatility_regime="calm",
            yield_curve_spread=-0.4,
            gold_usd_divergence=0.01,
            gold_momentum_5d=0.02,
            stale_age_seconds=12,
            is_stale=False,
        ),
    )
    return AnalysisBundle(
        question="今晚 CPI 超预期时，黄金短线如何控制风险？",
        optional_news_text=None,
        evidence_query="CPI 黄金 风险",
        horizon="24h",
        risk_profile={"label": "保守型", "description": "test"},
        investor_profile=None,
        risk_gate={"level": "low", "force_observation": False},
        snapshot=snapshot,
        forecast={"probability": 0.67},
        news=RecentNewsResponse(as_of=now, freshness_seconds=15, items=[]),
        rag_events=[],
        memory_status="ok",
        memory_degraded_reason=None,
        macro_context={"macro_signal": 1},
        news_sentiment=0.0,
        conflict_score=0,
        has_conflict=False,
        quant_probability=0.67,
        xgboost_probability=0.64,
        quant_direction=1,
        vix_value=18.0,
        is_high_risk=is_high_risk,
        is_low_confidence=False,
        has_degraded_inputs=False,
        degradation_flags=[],
        tool_trace=[],
    )


class _FakeDeepSeekCompletions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        if callable(output):
            return await output()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=output))]
        )


class _FakeDeepSeekClient:
    def __init__(self, outputs):
        self.completions = _FakeDeepSeekCompletions(outputs)
        self.chat = SimpleNamespace(completions=self.completions)


class _FakeOpenAIResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.outputs.pop(0))


class _FakeOpenAIClient:
    def __init__(self, outputs):
        self.responses = _FakeOpenAIResponses(outputs)


def _enhanced_json() -> str:
    data = _draft(action="小仓试探").model_dump()
    data["summary_card"]["disclaimer"] = "DeepSeek 增强"
    return json.dumps(data, ensure_ascii=False)


def test_deepseek_narrator_uses_json_mode_and_flash_model():
    client = _FakeDeepSeekClient([_enhanced_json()])
    narrator = OpenAINarrator(
        _config(),
        provider="deepseek",
        client=client,
        timeout_seconds=0.5,
    )

    result = asyncio.run(narrator.narrate(_bundle(), _draft()))

    assert result.summary_card.disclaimer == "DeepSeek 增强"
    assert len(client.completions.calls) == 1
    request = client.completions.calls[0]
    assert request["model"] == "deepseek-v4-flash"
    assert request["response_format"] == {"type": "json_object"}
    assert "json schema" in request["messages"][0]["content"].lower()


def test_deepseek_narrator_retries_invalid_json_once():
    client = _FakeDeepSeekClient(["not-json", _enhanced_json()])
    narrator = OpenAINarrator(
        _config(),
        provider="deepseek",
        client=client,
        timeout_seconds=0.5,
    )

    result = asyncio.run(narrator.narrate(_bundle(), _draft()))

    assert result.summary_card.disclaimer == "DeepSeek 增强"
    assert len(client.completions.calls) == 2


def test_deepseek_narrator_times_out_then_returns_draft():
    async def _slow_response():
        await asyncio.sleep(0.02)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=_enhanced_json()))]
        )

    client = _FakeDeepSeekClient([_slow_response, _slow_response])
    draft = _draft()
    narrator = OpenAINarrator(
        _config(),
        provider="deepseek",
        client=client,
        timeout_seconds=0.001,
    )

    result = asyncio.run(narrator.narrate(_bundle(), draft))

    assert result == draft
    assert len(client.completions.calls) == 2


def test_deepseek_narrator_retries_empty_output_then_returns_draft():
    client = _FakeDeepSeekClient(["", ""])
    draft = _draft()
    narrator = OpenAINarrator(
        _config(),
        provider="deepseek",
        client=client,
        timeout_seconds=0.5,
    )

    result = asyncio.run(narrator.narrate(_bundle(), draft))

    assert result == draft
    assert len(client.completions.calls) == 2


def test_deepseek_narrator_uses_pro_model_for_high_risk_bundle():
    client = _FakeDeepSeekClient([_enhanced_json()])
    narrator = OpenAINarrator(
        _config(),
        provider="deepseek",
        client=client,
        timeout_seconds=0.5,
    )

    asyncio.run(narrator.narrate(_bundle(is_high_risk=True), _draft()))

    assert client.completions.calls[0]["model"] == "deepseek-v4-pro"


def test_deepseek_narrator_without_key_returns_draft(monkeypatch, caplog):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    draft = _draft()
    narrator = OpenAINarrator(_config(), provider="deepseek")

    with caplog.at_level(logging.WARNING, logger="goldensense.agent_gateway"):
        result = asyncio.run(narrator.narrate(_bundle(), draft))

    assert result == draft
    assert "provider=deepseek model=deepseek-v4-flash reason=no_client_configured" in caplog.text


def test_openai_narrator_keeps_responses_api_adapter():
    client = _FakeOpenAIClient([_enhanced_json()])
    narrator = OpenAINarrator(
        _config(),
        provider="openai",
        client=client,
        timeout_seconds=0.5,
    )

    result = asyncio.run(narrator.narrate(_bundle(), _draft()))

    assert result.summary_card.disclaimer == "DeepSeek 增强"
    assert len(client.responses.calls) == 1
    assert client.responses.calls[0]["text"]["format"]["type"] == "json_schema"
