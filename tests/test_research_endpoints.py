"""Tests for the research/calibration endpoints and the committee wiring."""
import asyncio
import os
import threading
import traceback

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("AGENT_PUBLIC_API_KEYS", "dev-public-key")
os.environ.setdefault("AGENT_INTERNAL_API_KEYS", "dev-internal-key")

import agent_gateway  # noqa: E402
from signal_ledger import JsonlLedgerStore, MemoryLedgerStore  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app = agent_gateway.create_app()
    with TestClient(app) as test_client:
        yield test_client


PUBLIC_HEADERS = {"X-API-Key": "dev-public-key"}
LEDGER_UNAVAILABLE_PAYLOAD = {
    "status": "unavailable",
    "error_code": "signal_ledger_unavailable",
    "message": "信号发布台账暂时不可用，请稍后重试。",
    "guidance": "持久化恢复前不会返回空历史或接受新发布。",
    "retryable": True,
}


class _FaultingLedgerStore:
    def __init__(self, operation, sentinel):
        self._operation = operation
        self._sentinel = sentinel

    def load_all(self):
        if self._operation == "load":
            raise RuntimeError(f"store load failed: {self._sentinel}")
        return []

    def append(self, _record):
        if self._operation == "append":
            raise RuntimeError(f"store append failed: {self._sentinel}")


class _MissingConflictLedgerStore:
    def load_all(self):
        return []

    def append(self, _record):
        return False


class _HealthyTraceStore:
    def __init__(self):
        self.startup_thread_id = None

    async def startup(self):
        self.startup_thread_id = threading.get_ident()

    def health(self):
        return {"db_ready": True, "allow_memory_fallback": False}


def _configure_non_development(monkeypatch, app_env="production"):
    monkeypatch.setenv("APP_ENV", app_env)
    monkeypatch.setenv("AGENT_PUBLIC_API_KEYS", "production-public-key")
    monkeypatch.setenv("AGENT_INTERNAL_API_KEYS", "production-internal-key")
    monkeypatch.setenv("AGENT_WARM_QUANT_CONTEXT", "0")
    monkeypatch.setenv("SIGNAL_AUTOPUBLISH_ENABLED", "0")


def test_trace_store_startup_log_never_exposes_database_secret(monkeypatch, caplog):
    sentinel = "TRACE-DB-SECRET-9f3a"
    database_url = f"postgresql://trace-user:{sentinel}@db.example/traces"
    store = agent_gateway.AgentTraceStore(
        database_url,
        allow_memory_fallback=True,
        memory_ttl_seconds=60,
        memory_max_items=1,
    )

    def fail_schema_initialization():
        raise RuntimeError(f"connection rejected: {database_url}")

    monkeypatch.setattr(store, "_ensure_schema_sync", fail_schema_initialization)

    with caplog.at_level("WARNING"):
        asyncio.run(store.startup())

    assert "agent_trace_schema_init_failed" in caplog.text
    assert sentinel not in caplog.text
    assert database_url not in caplog.text


def test_trace_store_startup_exception_has_no_secret_bearing_chain(
    monkeypatch, caplog
):
    sentinel = "TRACE-PRODUCTION-SECRET-61ce"
    database_url = f"postgresql://trace-user:{sentinel}@db.example/traces"
    store = agent_gateway.AgentTraceStore(
        database_url,
        allow_memory_fallback=False,
        memory_ttl_seconds=60,
        memory_max_items=1,
    )

    def fail_schema_initialization():
        raise RuntimeError(f"connection rejected: {database_url}")

    monkeypatch.setattr(store, "_ensure_schema_sync", fail_schema_initialization)

    with caplog.at_level("WARNING"):
        with pytest.raises(agent_gateway.TraceStoreUnavailableError) as exc_info:
            asyncio.run(store.startup())

    formatted = "".join(
        traceback.format_exception(
            exc_info.type, exc_info.value, exc_info.tb, chain=True
        )
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in formatted
    assert database_url not in formatted
    assert sentinel not in caplog.text


@pytest.mark.parametrize(
    ("operation", "sync_method"),
    [
        ("persist", "_persist_analysis_sync"),
        ("feedback", "_save_feedback_sync"),
        ("load", "_load_trace_sync"),
        ("recent", "_load_recent_sync"),
    ],
)
def test_trace_store_runtime_exceptions_have_no_secret_bearing_chain(
    monkeypatch, caplog, operation, sync_method
):
    sentinel = "TRACE-RUNTIME-SECRET-a27f"
    database_url = f"postgresql://trace-user:{sentinel}@db.example/traces"
    store = agent_gateway.AgentTraceStore(
        database_url,
        allow_memory_fallback=False,
        memory_ttl_seconds=60,
        memory_max_items=1,
    )
    store._db_ready = True

    def fail_operation(*_args, **_kwargs):
        raise RuntimeError(f"database operation failed: {database_url}")

    monkeypatch.setattr(store, sync_method, fail_operation)

    async def invoke():
        if operation == "persist":
            return await store.persist_analysis(
                analysis_id="trace-1",
                request_payload={},
                tool_trace=[],
                evidence_payload={},
                response_payload={},
            )
        if operation == "feedback":
            return await store.save_feedback("trace-1", "helpful", None)
        if operation == "load":
            return await store.load_trace("trace-1")
        return await store.load_recent_analyses()

    with caplog.at_level("WARNING"):
        with pytest.raises(agent_gateway.TraceStoreUnavailableError) as exc_info:
            asyncio.run(invoke())

    formatted = "".join(
        traceback.format_exception(
            exc_info.type, exc_info.value, exc_info.tb, chain=True
        )
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in formatted
    assert database_url not in formatted
    assert sentinel not in caplog.text


def test_injected_signal_ledger_store_wins_and_preserves_empty_state(monkeypatch):
    _configure_non_development(monkeypatch)
    monkeypatch.setenv("SIGNAL_LEDGER_DATABASE_URL", "postgresql://must-not-be-used")
    injected = MemoryLedgerStore()

    def unexpected_postgres_store(_database_url):
        raise AssertionError("injected store must win")

    monkeypatch.setattr(
        agent_gateway, "PostgresLedgerStore", unexpected_postgres_store, raising=False
    )
    app = agent_gateway.create_app(
        trace_store=object(), signal_ledger_store=injected
    )
    with TestClient(app) as test_client:
        assert app.state.signal_ledger_store is injected
        response = test_client.get(
            "/api/v1/signals/current",
            headers={"X-API-Key": "production-public-key"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "empty"
    assert response.json()["error_code"] == "first_publication_required"


@pytest.mark.parametrize(
    ("app_env", "signal_url", "database_url", "expected_url"),
    [
        (
            "demo",
            "postgresql://signal-user:secret@signal-db/ledger",
            "postgresql://trace-user:secret@trace-db/traces",
            "postgresql://signal-user:secret@signal-db/ledger",
        ),
        (
            "production",
            None,
            "postgresql://shared-user:secret@shared-db/app",
            "postgresql://shared-user:secret@shared-db/app",
        ),
        (
            "staging",
            None,
            "postgresql://staging-user:secret@staging-db/app",
            "postgresql://staging-user:secret@staging-db/app",
        ),
    ],
)
def test_non_development_selects_postgres_from_explicit_url(
    monkeypatch, app_env, signal_url, database_url, expected_url
):
    _configure_non_development(monkeypatch, app_env)
    if signal_url is None:
        monkeypatch.delenv("SIGNAL_LEDGER_DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("SIGNAL_LEDGER_DATABASE_URL", signal_url)
    monkeypatch.setenv("DATABASE_URL", database_url)
    selected = MemoryLedgerStore()
    received_urls = []

    def fake_postgres_store(selected_url, **_kwargs):
        received_urls.append(selected_url)
        return selected

    monkeypatch.setattr(
        agent_gateway, "PostgresLedgerStore", fake_postgres_store, raising=False
    )
    app = agent_gateway.create_app(trace_store=object())
    with TestClient(app):
        assert app.state.signal_ledger_store is selected

    assert received_urls == [expected_url]


def test_explicit_development_without_database_url_retains_jsonl_store(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("SIGNAL_LEDGER_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SIGNAL_LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setenv("AGENT_WARM_QUANT_CONTEXT", "0")
    app = agent_gateway.create_app(trace_store=object())

    with TestClient(app):
        assert isinstance(app.state.signal_ledger_store, JsonlLedgerStore)


@pytest.mark.parametrize(
    "app_env", [None, "staging", "test", "unknown", "developmnt"]
)
def test_non_development_environment_without_database_url_fails_closed(
    monkeypatch, app_env
):
    monkeypatch.setenv("AGENT_PUBLIC_API_KEYS", "production-public-key")
    monkeypatch.setenv("AGENT_INTERNAL_API_KEYS", "production-internal-key")
    monkeypatch.setenv("AGENT_WARM_QUANT_CONTEXT", "0")
    monkeypatch.setenv("SIGNAL_AUTOPUBLISH_ENABLED", "0")
    monkeypatch.delenv("SIGNAL_LEDGER_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    if app_env is None:
        monkeypatch.delenv("APP_ENV", raising=False)
    else:
        monkeypatch.setenv("APP_ENV", app_env)
    app = agent_gateway.create_app(trace_store=object())

    with TestClient(app) as test_client:
        response = test_client.get(
            "/api/v1/signals/current",
            headers={"X-API-Key": "production-public-key"},
        )
        assert app.state.signal_ledger_available is False
        assert not isinstance(app.state.signal_ledger_store, JsonlLedgerStore)

    assert response.status_code == 503
    assert response.json() == LEDGER_UNAVAILABLE_PAYLOAD


def test_production_ledger_initialization_failure_is_typed_and_sanitized(
    monkeypatch, caplog
):
    _configure_non_development(monkeypatch)
    secret_url = "postgresql://ledger-user:super-secret@db.example/ledger"
    monkeypatch.setenv("SIGNAL_LEDGER_DATABASE_URL", secret_url)

    def failing_postgres_store(database_url):
        raise RuntimeError(f"connection failed for {database_url}")

    monkeypatch.setattr(
        agent_gateway, "PostgresLedgerStore", failing_postgres_store, raising=False
    )
    app = agent_gateway.create_app(trace_store=object())
    public_headers = {"X-API-Key": "production-public-key"}
    internal_headers = {"X-API-Key": "production-internal-key"}
    requests = [
        ("get", "/api/v1/signals/current", public_headers),
        ("get", "/api/v1/signals/history", public_headers),
        ("get", "/api/v1/signals/track-record", public_headers),
        ("post", "/api/v1/signals/publish", internal_headers),
    ]

    with caplog.at_level("ERROR"), TestClient(app) as test_client:
        responses = [
            getattr(test_client, method)(path, headers=headers)
            for method, path, headers in requests
        ]

    assert all(response.status_code == 503 for response in responses)
    assert all(response.json() == LEDGER_UNAVAILABLE_PAYLOAD for response in responses)
    assert "signal_ledger_initialization_failed" in caplog.text
    assert secret_url not in caplog.text
    assert "super-secret" not in caplog.text


def test_production_without_database_url_never_invents_localhost_ledger(
    monkeypatch, caplog
):
    _configure_non_development(monkeypatch)
    monkeypatch.delenv("SIGNAL_LEDGER_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    received_urls = []

    def fake_postgres_store(database_url):
        received_urls.append(database_url)
        return MemoryLedgerStore()

    monkeypatch.setattr(
        agent_gateway, "PostgresLedgerStore", fake_postgres_store, raising=False
    )
    app = agent_gateway.create_app(trace_store=object())

    with caplog.at_level("ERROR"), TestClient(app) as test_client:
        response = test_client.get(
            "/api/v1/signals/current",
            headers={"X-API-Key": "production-public-key"},
        )

    assert received_urls == []
    assert response.status_code == 503
    assert response.json()["error_code"] == "signal_ledger_unavailable"
    assert "localhost" not in caplog.text


def test_readiness_fails_when_signal_ledger_is_unavailable(monkeypatch):
    _configure_non_development(monkeypatch)
    monkeypatch.delenv("SIGNAL_LEDGER_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    trace_store = _HealthyTraceStore()
    app = agent_gateway.create_app(toolbox=object(), trace_store=trace_store)

    with TestClient(app) as test_client:
        response = test_client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert "signal_ledger_unavailable" in response.json()["errors"]
    assert response.json()["signal_ledger"]["status"] == "unavailable"


def test_runtime_ledger_failure_updates_readiness_state(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    trace_store = _HealthyTraceStore()
    app = agent_gateway.create_app(
        toolbox=object(),
        trace_store=trace_store,
        signal_ledger_store=_FaultingLedgerStore("load", "runtime-outage"),
    )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        ledger_response = test_client.get(
            "/api/v1/signals/current", headers=PUBLIC_HEADERS
        )
        ready_response = test_client.get("/health/ready")

    assert ledger_response.status_code == 503
    assert ready_response.status_code == 503
    assert "signal_ledger_unavailable" in ready_response.json()["errors"]
    assert ready_response.json()["signal_ledger"]["status"] == "unavailable"


def test_readiness_probes_ledger_without_waiting_for_public_request(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    trace_store = _HealthyTraceStore()
    app = agent_gateway.create_app(
        toolbox=object(),
        trace_store=trace_store,
        signal_ledger_store=_FaultingLedgerStore("load", "runtime-outage"),
    )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        ready_response = test_client.get("/health/ready")

    assert ready_response.status_code == 503
    assert "signal_ledger_unavailable" in ready_response.json()["errors"]
    assert ready_response.json()["signal_ledger"]["status"] == "unavailable"


def test_readiness_reprobes_and_recovers_after_transient_ledger_failure(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")

    class RecoveringStore(MemoryLedgerStore):
        def __init__(self):
            super().__init__()
            self.health_calls = 0

        def healthcheck(self):
            self.health_calls += 1
            if self.health_calls == 1:
                raise RuntimeError("temporary outage")
            return True

    store = RecoveringStore()
    app = agent_gateway.create_app(
        toolbox=object(),
        trace_store=_HealthyTraceStore(),
        signal_ledger_store=store,
    )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        first = test_client.get("/health/ready")
        second = test_client.get("/health/ready")

    assert first.status_code == 503
    assert second.status_code == 200
    assert second.json()["signal_ledger"]["status"] == "ok"
    assert store.health_calls == 2


def test_autopublish_completes_before_app_accepts_requests(monkeypatch):
    from test_signal_ledger import _ctx

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("SIGNAL_AUTOPUBLISH_ENABLED", "1")
    monkeypatch.setattr(agent_gateway.quant_research_context, "get_context", _ctx)
    release_append = threading.Event()

    class DelayedStore(MemoryLedgerStore):
        def append(self, record):
            assert release_append.wait(timeout=2)
            return super().append(record)

    store = DelayedStore()
    app = agent_gateway.create_app(
        toolbox=object(),
        trace_store=object(),
        signal_ledger_store=store,
    )

    async def enter_lifespan():
        asyncio.get_running_loop().call_later(0.05, release_append.set)
        async with app.router.lifespan_context(app):
            return store.load_all()

    records_at_accept = asyncio.run(enter_lifespan())

    assert len(records_at_accept) == 1
    assert records_at_accept[0]["data_asof"] == "2026-07-13"



def test_autopublish_with_stale_research_holds_without_blocking_startup(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("SIGNAL_AUTOPUBLISH_ENABLED", "1")
    monkeypatch.setattr(agent_gateway.quant_research_context, "get_context", lambda: {"data_stale": True})
    store = MemoryLedgerStore()
    app = agent_gateway.create_app(toolbox=object(), trace_store=object(), signal_ledger_store=store)

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert store.load_all() == []

def test_autopublish_failure_blocks_startup(monkeypatch):
    from test_signal_ledger import _ctx

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("SIGNAL_AUTOPUBLISH_ENABLED", "1")
    monkeypatch.setattr(agent_gateway.quant_research_context, "get_context", _ctx)
    app = agent_gateway.create_app(
        toolbox=object(),
        trace_store=object(),
        signal_ledger_store=_FaultingLedgerStore("append", "secret-value"),
    )

    with pytest.raises(RuntimeError, match="Initial signal publication failed"):
        with TestClient(app):
            pass


def test_postgres_ledger_initialization_is_offloaded_and_bounded(monkeypatch):
    _configure_non_development(monkeypatch)
    monkeypatch.setenv(
        "SIGNAL_LEDGER_DATABASE_URL", "postgresql://ledger.example/app"
    )
    monkeypatch.setenv("SIGNAL_LEDGER_CONNECT_TIMEOUT_SECONDS", "3")
    constructor_calls = []
    selected = MemoryLedgerStore()

    def fake_postgres_store(database_url, **kwargs):
        constructor_calls.append(
            {
                "database_url": database_url,
                "kwargs": kwargs,
                "thread_id": threading.get_ident(),
            }
        )
        return selected

    monkeypatch.setattr(agent_gateway, "PostgresLedgerStore", fake_postgres_store)
    trace_store = _HealthyTraceStore()
    app = agent_gateway.create_app(toolbox=object(), trace_store=trace_store)

    with TestClient(app):
        assert app.state.signal_ledger_store is selected

    assert constructor_calls == [
        {
            "database_url": "postgresql://ledger.example/app",
            "kwargs": {"connect_timeout_seconds": 3},
            "thread_id": constructor_calls[0]["thread_id"],
        }
    ]
    assert constructor_calls[0]["thread_id"] != trace_store.startup_thread_id


@pytest.mark.parametrize(
    ("method", "path", "headers", "fault_operation"),
    [
        ("get", "/api/v1/signals/current", PUBLIC_HEADERS, "load"),
        ("get", "/api/v1/signals/history", PUBLIC_HEADERS, "load"),
        ("get", "/api/v1/signals/track-record", PUBLIC_HEADERS, "load"),
        (
            "post",
            "/api/v1/signals/publish",
            {"X-API-Key": "dev-internal-key"},
            "append",
        ),
    ],
)
def test_runtime_store_failures_use_typed_safe_unavailable_response(
    monkeypatch, caplog, method, path, headers, fault_operation
):
    sentinel = "RUNTIME-LEDGER-SECRET-4b7d"
    monkeypatch.setenv("APP_ENV", "development")
    if fault_operation == "append":
        from test_signal_ledger import _ctx

        monkeypatch.setattr(agent_gateway.quant_research_context, "get_context", _ctx)
    app = agent_gateway.create_app(
        trace_store=object(),
        signal_ledger_store=_FaultingLedgerStore(fault_operation, sentinel),
    )

    with caplog.at_level("ERROR"), TestClient(
        app, raise_server_exceptions=False
    ) as test_client:
        response = getattr(test_client, method)(path, headers=headers)

    assert response.status_code == 503
    assert response.json() == LEDGER_UNAVAILABLE_PAYLOAD
    assert sentinel not in response.text
    assert sentinel not in caplog.text


@pytest.mark.parametrize(
    ("method", "path", "headers", "failure_target", "expected_code"),
    [
        (
            "get",
            "/api/v1/signals/track-record",
            PUBLIC_HEADERS,
            "market_data",
            "track_record_unavailable",
        ),
        (
            "post",
            "/api/v1/signals/publish",
            {"X-API-Key": "dev-internal-key"},
            "research_context",
            "publish_failed",
        ),
    ],
)
def test_non_ledger_signal_failures_are_sanitized(
    monkeypatch, caplog, method, path, headers, failure_target, expected_code
):
    import data_sources

    sentinel = "NON-LEDGER-SECRET-DSN-7f31"
    secret_url = f"postgresql://user:{sentinel}@db.example/app"
    monkeypatch.setenv("APP_ENV", "development")

    def fail_with_secret(*_args, **_kwargs):
        raise RuntimeError(f"upstream failed at {secret_url}")

    if failure_target == "market_data":
        monkeypatch.setattr(data_sources, "load_market_data", fail_with_secret)
    else:
        monkeypatch.setattr(
            agent_gateway.quant_research_context,
            "get_context",
            fail_with_secret,
        )

    app = agent_gateway.create_app(
        trace_store=object(), signal_ledger_store=MemoryLedgerStore()
    )
    with caplog.at_level("ERROR"), TestClient(
        app, raise_server_exceptions=False
    ) as test_client:
        response = getattr(test_client, method)(path, headers=headers)

    assert response.status_code == 503
    assert response.json()["detail"]["error_code"] == expected_code
    assert sentinel not in response.text
    assert secret_url not in response.text
    assert sentinel not in caplog.text


def test_guarded_ledger_exception_has_no_secret_bearing_chain():
    sentinel = "GUARDED-LEDGER-SECRET-c391"
    guarded = agent_gateway._GuardedLedgerStore(
        _FaultingLedgerStore("load", sentinel)
    )

    with pytest.raises(agent_gateway.LedgerUnavailableError) as exc_info:
        guarded.load_all()

    formatted = "".join(
        traceback.format_exception(
            exc_info.type, exc_info.value, exc_info.tb, chain=True
        )
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in formatted


def test_publish_conflict_without_readable_record_is_typed_ledger_unavailable(
    monkeypatch
):
    from test_signal_ledger import _ctx

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setattr(agent_gateway.quant_research_context, "get_context", _ctx)
    app = agent_gateway.create_app(
        trace_store=object(), signal_ledger_store=_MissingConflictLedgerStore()
    )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post(
            "/api/v1/signals/publish",
            headers={"X-API-Key": "dev-internal-key"},
        )

    assert response.status_code == 503
    assert response.json() == LEDGER_UNAVAILABLE_PAYLOAD
    assert "LedgerIntegrityError" not in response.text
    assert "Conflicting signal publication" not in response.text


def test_research_current_requires_api_key(client):
    resp = client.get("/api/v1/agent/research/current")
    assert resp.status_code in (401, 403)


def test_research_current_returns_quant_blocks(client):
    resp = client.get("/api/v1/agent/research/current", headers=PUBLIC_HEADERS)
    assert resp.status_code == 200
    payload = resp.json()
    # Every block is either present or explicitly listed as degraded.
    expected_blocks = {
        "regime_posterior",
        "fair_value",
        "vol_bands",
        "macro_factors",
        "scenario_cone",
        "allocation",
    }
    for block in expected_blocks:
        assert block in payload or block in payload.get("degraded", {}), block
    assert "data_source" in payload or "market_data" in payload.get("degraded", {})


def test_research_current_reports_data_freshness(client):
    payload = client.get(
        "/api/v1/agent/research/current", headers=PUBLIC_HEADERS
    ).json()
    if "market_data" in payload.get("degraded", {}):
        pytest.skip("market data degraded in this environment")
    # The quant page must never imply real-time data; freshness must be explicit.
    assert payload["is_realtime"] is False
    assert "data_asof" in payload
    assert isinstance(payload["data_age_days"], int)
    assert isinstance(payload["data_stale"], bool)
    assert payload["data_source"] in {"extended", "base"}


def test_research_current_includes_flagship_backtest(client):
    payload = client.get(
        "/api/v1/agent/research/current", headers=PUBLIC_HEADERS
    ).json()
    flagship = payload.get("flagship")
    if flagship is None:
        assert "flagship" in payload.get("degraded", {})
        return
    for key in ("sharpe", "sortino", "max_drawdown", "calmar", "dsr"):
        assert key in flagship["metrics"]
        assert key in flagship["benchmark"]
    curve = flagship["curve"]
    assert len(curve["dates"]) == len(curve["flagship"]) == len(curve["benchmark"])
    assert curve["flagship"] and curve["benchmark"]
    # Drawdown-aware sleeve: the flagship's max drawdown is shallower than B&H.
    assert flagship["metrics"]["max_drawdown"] > flagship["benchmark"]["max_drawdown"]
    assert 0.0 <= flagship["metrics"]["dsr"] <= 1.0


def test_research_current_regime_posterior_shape(client):
    payload = client.get(
        "/api/v1/agent/research/current", headers=PUBLIC_HEADERS
    ).json()
    regime = payload.get("regime_posterior")
    if regime is None:
        pytest.skip("regime posterior degraded in this environment")
    latest = regime["latest"]
    assert set(latest) == {"calm", "elevated", "stress"}
    assert abs(sum(latest.values()) - 1.0) < 0.02


def test_calibration_endpoint_contract(client):
    resp = client.get("/api/v1/agent/calibration", headers=PUBLIC_HEADERS)
    assert resp.status_code == 200
    payload = resp.json()
    for key in (
        "total_scored",
        "directional_calls",
        "neutral_or_gated",
        "minimum_directional_calls",
        "pending_analyses",
        "evidence_status",
        "weight_adjustment",
        "recent_outcomes",
        "governance",
    ):
        assert key in payload
    assert payload["minimum_directional_calls"] == 20
    assert payload["pending_analyses"] >= 0
    assert payload["evidence_status"] in {
        "awaiting_first_analysis",
        "accumulating",
        "ready",
    }
    if payload["directional_calls"] < payload["minimum_directional_calls"]:
        assert payload["hit_rate"] is None
        assert payload["brier_score"] is None
    assert payload["weight_adjustment"]["fused_confidence_multiplier"] >= 0.8
    gov = payload["governance"]
    assert gov["mode"] in {"champion", "watch", "demoted", "insufficient_data"}
    assert isinstance(gov["demoted"], bool)
    # demoted is only true in the demoted mode (env may or may not have matured
    # rows in a shared trace DB, so don't assert a specific mode here).
    assert gov["demoted"] == (gov["mode"] == "demoted")
    assert 0.8 <= gov["confidence_multiplier"] <= 1.2


def test_analyze_response_trace_contains_committee():
    # The analyze path needs downstream tools; reuse the scenario toolbox
    # from the main analyze test-suite instead of live services.
    from test_agent_analyze import _DraftNarrator, _ScenarioToolbox

    app = agent_gateway.create_app(toolbox=_ScenarioToolbox(), narrator=_DraftNarrator())
    with TestClient(app) as client:
        _assert_committee_in_trace(client)


def test_long_term_research_uses_valuation_view_independently_of_forecast_window(monkeypatch):
    from test_agent_analyze import _DraftNarrator, _ScenarioToolbox

    monkeypatch.setattr(
        agent_gateway.quant_research_context, "get_context",
        lambda: {"fair_value": {"deviation_pct": 8.0, "deviation_z": 0.5}},
    )
    app = agent_gateway.create_app(toolbox=_ScenarioToolbox(), narrator=_DraftNarrator())
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/analyze", headers=PUBLIC_HEADERS,
            json={"question": "黄金长期估值怎么看？", "horizon": "long_term"},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["research_focus"]["horizon"] == "long_term"
    assert "估值" in result["research_focus"]["section"]["core_view"]
    assert result["summary_card"]["horizon"] == "long_term"
    assert result["horizon_forecasts"][-1]["horizon"] == "long_term"


def _assert_committee_in_trace(client):
    resp = client.post(
        "/api/v1/agent/analyze",
        headers=PUBLIC_HEADERS,
        json={
            "question": "现在黄金的中期趋势怎么看？",
            "risk_profile": "balanced",
            "horizon": "mid_term",
            "locale": "zh-CN",
        },
    )
    assert resp.status_code == 200
    analysis_id = resp.json()["analysis_id"]

    trace = client.get(
        f"/api/v1/agent/traces/{analysis_id}",
        headers={"X-API-Key": "dev-internal-key"},
    )
    assert trace.status_code == 200
    bundle = trace.json()["evidence_payload"]["bundle"]
    assert "committee" in bundle
    committee = bundle["committee"]
    assert {v["name"] for v in committee["views"]} == {
        "technical",
        "macro",
        "flow",
        "news",
    }
    assert -1.0 <= committee["fused_stance"] <= 1.0
    assert 0.0 <= committee["disagreement"] <= 1.0
    assert "regime" in bundle

    # Governance overlay must be recorded in the tool trace (champion on a
    # fresh store -> no demotion flag).
    tool_trace = trace.json()["tool_trace"]
    gov_entries = [t for t in tool_trace if t.get("tool") == "model_governance"]
    assert gov_entries, "model_governance trace entry missing"
    verdict = gov_entries[0]["verdict"]
    assert verdict["mode"] in {"champion", "watch", "demoted", "insufficient_data"}
    # The demotion flag must appear in degradation_flags iff governance demoted.
    flags = trace.json()["response_payload"]["degradation_flags"]
    assert ("model_demoted_by_performance" in flags) == verdict["demoted"]
