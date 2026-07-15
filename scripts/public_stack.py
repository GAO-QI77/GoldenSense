from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    command: tuple[str, ...]
    env: dict[str, str]


def _service_env(source: Mapping[str, str], defaults: Mapping[str, str]) -> dict[str, str]:
    env = dict(source)
    for key, value in defaults.items():
        env.setdefault(key, value)
    return env


def _uvicorn(module: str, *, host: str, port: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "uvicorn",
        f"{module}:app",
        "--host",
        host,
        "--port",
        port,
    )


def validate_public_env(env: Mapping[str, str]) -> None:
    if env.get("APP_ENV", "demo").lower() == "development":
        raise RuntimeError("APP_ENV=development is not allowed in the public stack.")
    public_keys = env.get("AGENT_PUBLIC_API_KEYS", "")
    internal_keys = env.get("AGENT_INTERNAL_API_KEYS", "")
    if not public_keys or not internal_keys:
        raise RuntimeError("AGENT_PUBLIC_API_KEYS and AGENT_INTERNAL_API_KEYS are required for the public stack.")
    if "dev-public-key" in public_keys.split(",") or "dev-internal-key" in internal_keys.split(","):
        raise RuntimeError("Default development API keys are not allowed in the public stack.")


def build_service_specs(source_env: Mapping[str, str]) -> list[ServiceSpec]:
    env = _service_env(
        source_env,
        {
            "APP_ENV": "demo",
            "DATABASE_URL": "postgresql://localhost/postgres",
            "REDIS_URL": "redis://localhost:6379/0",
        },
    )
    public_port = env.get("PORT", "8020")
    inference = ServiceSpec(
        name="inference",
        command=_uvicorn("inference_service", host="127.0.0.1", port="8010"),
        env=_service_env(
            env,
            {
                "INFERENCE_MODEL_CHECKPOINTS_DIR_T1": "model_checkpoints",
                "INFERENCE_MODEL_CHECKPOINTS_DIR_T7": "model_checkpoints",
                "INFERENCE_ALLOW_SYNTHETIC_FALLBACK": "1",
            },
        ),
    )
    memory = ServiceSpec(
        name="memory",
        command=_uvicorn("memory_service", host="127.0.0.1", port="8012"),
        env=_service_env(
            env,
            {
                "MEMORY_START_BACKGROUND_LOAD": "0",
                "MEMORY_ALLOW_UNAVAILABLE_READY": "1",
            },
        ),
    )
    market = ServiceSpec(
        name="market",
        command=_uvicorn("market_snapshot_service", host="127.0.0.1", port="8014"),
        env=_service_env(
            env,
            {
                "MARKET_DATA_PROVIDER": "yfinance",
                "MARKET_START_BACKGROUND_TASK": "1",
                "MARKET_ALLOW_SYNTHETIC_FALLBACK": "1",
            },
        ),
    )
    news = ServiceSpec(
        name="news",
        command=_uvicorn("news_ingest_service", host="127.0.0.1", port="8016"),
        env=_service_env(
            env,
            {
                "NEWS_DATA_PROVIDER": "rss",
                "NEWS_START_BACKGROUND_TASK": "1",
                "NEWS_ALLOW_SAMPLE_FALLBACK": "1",
            },
        ),
    )
    agent = ServiceSpec(
        name="agent",
        command=_uvicorn("agent_gateway", host="0.0.0.0", port=public_port),
        env=_service_env(
            env,
            {
                "FORECAST_URL": "http://127.0.0.1:8010/api/v1/forecast",
                "MEMORY_URL": "http://127.0.0.1:8012/api/v1/memory/search",
                "MARKET_SNAPSHOT_URL": "http://127.0.0.1:8014/api/v1/market/snapshot/latest",
                "MARKET_INDICATORS_URL": "http://127.0.0.1:8014/api/v1/market/indicators/current",
                "MARKET_HISTORY_URL": "http://127.0.0.1:8014/api/v1/market/gold/history",
                "RECENT_NEWS_URL": "http://127.0.0.1:8016/api/v1/news/recent",
                "AGENT_ALLOW_TRACE_MEMORY_FALLBACK": "1",
                "AGENT_ANALYZE_RATE_LIMIT_PER_MINUTE": "20",
            },
        ),
    )
    return [inference, memory, market, news, agent]


def _stop_all(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 8.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    env = dict(os.environ)
    env.setdefault("APP_ENV", "demo")
    validate_public_env(env)
    specs = build_service_specs(env)
    processes: list[subprocess.Popen[bytes]] = []

    def _handle_signal(_signum: int, _frame: object) -> None:
        _stop_all(processes)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    for spec in specs:
        print(f"starting {spec.name}: {' '.join(spec.command)}", flush=True)
        processes.append(subprocess.Popen(spec.command, env=spec.env))

    try:
        while True:
            for spec, process in zip(specs, processes):
                return_code = process.poll()
                if return_code is not None:
                    print(f"{spec.name} exited with status {return_code}", file=sys.stderr, flush=True)
                    return return_code or 1
            time.sleep(1.0)
    finally:
        _stop_all(processes)


if __name__ == "__main__":
    raise SystemExit(main())
