from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import public_stack
from scripts.public_stack import ServiceSpec, build_service_specs, validate_public_env


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
    assert by_name["agent"].env["SIGNAL_AUTOPUBLISH_ENABLED"] == "1"
    assert by_name["agent"].env["DATA_REFRESH_ENABLED"] == "1"


def test_public_stack_never_invents_a_local_database_url():
    specs = build_service_specs(_base_env())

    assert all("DATABASE_URL" not in spec.env for spec in specs)


def test_public_stack_respects_explicit_refresh_and_autopublish_overrides():
    env = {
        **_base_env(),
        "DATA_REFRESH_ENABLED": "0",
        "SIGNAL_AUTOPUBLISH_ENABLED": "0",
    }
    specs = build_service_specs(env)
    spawned = []

    public_stack.start_public_services(
        specs,
        env=env,
        refresh_once_fn=lambda: pytest.fail("refresh must remain disabled"),
        start_background_refresh_fn=lambda **_kwargs: pytest.fail(
            "scheduler must remain disabled"
        ),
        popen_fn=lambda command, *, env: spawned.append((command, env)),
    )

    assert len(spawned) == len(specs)
    assert all(spec.env["DATA_REFRESH_ENABLED"] == "0" for spec in specs)
    assert {spec.name: spec for spec in specs}["agent"].env[
        "SIGNAL_AUTOPUBLISH_ENABLED"
    ] == "0"


def test_public_stack_refreshes_then_schedules_then_starts_services():
    events = []
    specs = [
        ServiceSpec("inference", ("inference",), {"SERVICE": "inference"}),
        ServiceSpec("agent", ("agent",), {"SERVICE": "agent"}),
    ]

    def refresh_once():
        events.append(("refresh",))
        return {
            "ok": True,
            "step": "dataset_built",
            "data_asof": "2026-09-05",
            "data_source": "extended",
        }

    def start_background_refresh(*, run_immediately, enabled):
        events.append(("scheduler", run_immediately, enabled))
        return object()

    def popen(command, *, env):
        events.append(("spawn", command[0], env["SERVICE"]))
        return object()

    processes = public_stack.start_public_services(
        specs,
        env={"DATA_REFRESH_ENABLED": "1"},
        refresh_once_fn=refresh_once,
        start_background_refresh_fn=start_background_refresh,
        popen_fn=popen,
    )

    assert events == [
        ("refresh",),
        ("scheduler", False, True),
        ("spawn", "inference", "inference"),
        ("spawn", "agent", "agent"),
    ]
    assert len(processes) == 2


def test_public_stack_cleans_up_children_when_a_later_service_fails(monkeypatch):
    first_process = object()
    stopped = []
    attempts = 0

    def popen(_command, *, env):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return first_process
        raise OSError("second service failed to start")

    monkeypatch.setattr(
        public_stack,
        "_stop_all",
        lambda processes: stopped.extend(processes),
    )

    with pytest.raises(OSError, match="second service failed"):
        public_stack.start_public_services(
            [
                ServiceSpec("inference", ("inference",), {}),
                ServiceSpec("agent", ("agent",), {}),
            ],
            env={"DATA_REFRESH_ENABLED": "0"},
            popen_fn=popen,
        )

    assert stopped == [first_process]


def test_public_stack_starts_with_explicit_last_good_degradation(capsys):
    spawned = []
    secret_url = "https://provider.example/data?api_key=secret-value"

    public_stack.start_public_services(
        [ServiceSpec("agent", ("agent",), {})],
        env={"DATA_REFRESH_ENABLED": "1"},
        refresh_once_fn=lambda: {
            "ok": False,
            "step": "dataset_failed_kept_last_good",
            "dataset_error": f"provider failed at {secret_url}",
            "data_asof": "2026-09-04",
            "data_source": "extended",
        },
        start_background_refresh_fn=lambda *, run_immediately, enabled: object(),
        popen_fn=lambda command, *, env: spawned.append(command),
    )

    output = capsys.readouterr().out
    assert spawned == [("agent",)]
    assert '"status": "last_good"' in output
    assert secret_url not in output
    assert "secret-value" not in output


def test_public_stack_refuses_to_start_without_usable_refresh_context(capsys):
    events = []
    secret_url = "https://provider.example/data?api_key=secret-value"

    with pytest.raises(RuntimeError, match="no usable data_asof"):
        public_stack.start_public_services(
            [ServiceSpec("agent", ("agent",), {})],
            env={"DATA_REFRESH_ENABLED": "1"},
            refresh_once_fn=lambda: {
                "ok": False,
                "step": "dataset_failed_kept_last_good",
                "context_error": f"provider failed at {secret_url}",
            },
            start_background_refresh_fn=lambda **_kwargs: events.append("scheduler"),
            popen_fn=lambda *_args, **_kwargs: events.append("spawn"),
        )

    output = capsys.readouterr().out
    assert events == []
    assert '"status": "unavailable"' in output
    assert secret_url not in output
    assert "secret-value" not in output


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


@pytest.mark.parametrize(
    "app_env",
    ["development", " Development ", "DEVELOPMENT"],
)
def test_public_stack_rejects_development_mode(app_env):
    with pytest.raises(RuntimeError, match="APP_ENV"):
        validate_public_env(
            {
                "APP_ENV": app_env,
                "AGENT_PUBLIC_API_KEYS": "public-demo-key",
                "AGENT_INTERNAL_API_KEYS": "internal-demo-key",
            }
        )


def test_railway_deployment_probes_full_readiness():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "railway.json").read_text()
    )

    assert config["deploy"]["healthcheckPath"] == "/health/ready"
