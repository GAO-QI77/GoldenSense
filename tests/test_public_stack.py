from __future__ import annotations

import pytest

from scripts.public_stack import build_service_specs, validate_public_env


def _base_env() -> dict[str, str]:
    return {
        "APP_ENV": "demo",
        "AGENT_PUBLIC_API_KEYS": "public-demo-key",
        "AGENT_INTERNAL_API_KEYS": "internal-demo-key",
        "PORT": "9000",
    }


def test_public_stack_only_exposes_agent_gateway():
    specs = build_service_specs(_base_env())
    by_name = {spec.name: spec for spec in specs}

    assert set(by_name) == {"inference", "memory", "market", "news", "agent"}
    for name in {"inference", "memory", "market", "news"}:
        assert "--host" in by_name[name].command
        assert by_name[name].command[by_name[name].command.index("--host") + 1] == "127.0.0.1"

    gateway = by_name["agent"]
    assert gateway.command[gateway.command.index("--host") + 1] == "0.0.0.0"
    assert gateway.command[gateway.command.index("--port") + 1] == "9000"


def test_public_stack_demo_defaults_keep_free_sources_and_explicit_fallbacks():
    specs = build_service_specs(_base_env())
    by_name = {spec.name: spec for spec in specs}

    assert by_name["market"].env["MARKET_DATA_PROVIDER"] == "yfinance"
    assert by_name["market"].env["MARKET_ALLOW_SYNTHETIC_FALLBACK"] == "1"
    assert by_name["news"].env["NEWS_DATA_PROVIDER"] == "rss"
    assert by_name["news"].env["NEWS_ALLOW_SAMPLE_FALLBACK"] == "1"
    assert by_name["memory"].env["MEMORY_START_BACKGROUND_LOAD"] == "0"
    assert by_name["memory"].env["MEMORY_ALLOW_UNAVAILABLE_READY"] == "1"
    assert by_name["agent"].env["AGENT_ALLOW_TRACE_MEMORY_FALLBACK"] == "1"
    assert by_name["agent"].env["AGENT_ANALYZE_RATE_LIMIT_PER_MINUTE"] == "20"


def test_public_stack_rejects_missing_or_default_api_keys():
    with pytest.raises(RuntimeError, match="AGENT_PUBLIC_API_KEYS"):
        validate_public_env({"APP_ENV": "demo"})

    with pytest.raises(RuntimeError, match="development API keys"):
        validate_public_env(
            {
                "APP_ENV": "demo",
                "AGENT_PUBLIC_API_KEYS": "dev-public-key",
                "AGENT_INTERNAL_API_KEYS": "dev-internal-key",
            }
        )


def test_public_stack_rejects_development_mode():
    with pytest.raises(RuntimeError, match="APP_ENV"):
        validate_public_env(
            {
                "APP_ENV": "development",
                "AGENT_PUBLIC_API_KEYS": "public-demo-key",
                "AGENT_INTERNAL_API_KEYS": "internal-demo-key",
            }
        )
