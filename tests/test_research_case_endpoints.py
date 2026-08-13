import os
from io import BytesIO

import httpx
from fastapi.testclient import TestClient
from PIL import Image
from reportlab.pdfgen import canvas

os.environ.setdefault("AGENT_PUBLIC_API_KEYS", "dev-public-key")
os.environ.setdefault("AGENT_INTERNAL_API_KEYS", "dev-internal-key")

import agent_gateway  # noqa: E402
from research_case import ResearchCaseStore  # noqa: E402
from signal_ledger import MemoryLedgerStore  # noqa: E402

HEADERS = {"X-API-Key": "dev-public-key"}


def _ctx():
    return {
        "data_asof": "2026-08-12", "data_age_days": 1, "data_stale": False,
        "data_source": "fixture", "is_realtime": False,
        "vol_bands": {
            "h1": {"p10": -0.01, "p50": 0.0, "p90": 0.01},
            "h5": {"p10": -0.02, "p50": 0.0, "p90": 0.02},
            "h21": {"p10": -0.05, "p50": 0.01, "p90": 0.06, "ann_vol_forecast": 0.18},
        },
        "regime_posterior": {"latest": {"calm": 0.2, "elevated": 0.7, "stress": 0.1}},
        "macro_factors": {"composite": 0.6, "factors_used": ["real_rate_momentum"]},
        "fair_value": {"deviation_pct": 6.0, "deviation_z": 1.0, "regime_break": False},
        "scenario_cone": {"checkpoints": {"d90": {"p10": 3100, "p50": 3400, "p90": 3700}}},
        "allocation": {
            "conservative": {"reference_range_pct": [2, 8]},
            "balanced": {"reference_range_pct": [5, 15]},
            "aggressive": {"reference_range_pct": [10, 25]},
        },
        "flagship": {
            "metrics": {"sharpe": 0.7, "max_drawdown": -0.21},
            "governance": {"mode": "champion", "reason": "validated"},
            "sample": {"start": "2004-01-01", "end": "2026-07-14"},
        },
        "degraded": {},
    }


def _pdf(text):
    output = BytesIO()
    document = canvas.Canvas(output)
    document.drawString(72, 760, text)
    document.save()
    return output.getvalue()


def _png():
    output = BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, "PNG")
    return output.getvalue()


def _client(monkeypatch, *, http=None):
    monkeypatch.setattr(agent_gateway.quant_research_context, "get_context", lambda: _ctx())
    app = agent_gateway.create_app(
        signal_ledger_store=MemoryLedgerStore(),
        research_case_store=ResearchCaseStore(max_items=20),
        http_client=http,
    )
    return TestClient(app)


def test_create_requires_api_key(monkeypatch):
    with _client(monkeypatch) as client:
        assert client.post("/api/v1/agent/research-cases", json={"question": "gold"}).status_code in (401, 403)


def test_question_only_case_can_be_created_and_retrieved(monkeypatch):
    with _client(monkeypatch) as client:
        created = client.post(
            "/api/v1/agent/research-cases", headers=HEADERS,
            json={"question": "What is the medium-term gold regime?", "mode": "draft"},
        )
        assert created.status_code == 201
        payload = created.json()
        assert payload["case_id"].startswith("rc_")
        assert payload["status"] == "degraded"
        fetched = client.get(
            f"/api/v1/agent/research-cases/{payload['case_id']}", headers=HEADERS
        )
        assert fetched.status_code == 200
        assert fetched.json()["data_asof"] == "2026-08-12"


def test_pdf_and_image_multipart_inputs_are_real_paths(monkeypatch):
    with _client(monkeypatch) as client:
        pdf = client.post(
            "/api/v1/agent/research-cases", headers=HEADERS,
            data={"question": "FOMC impact", "mode": "full"},
            files={"file": ("fomc.pdf", _pdf("Fed held rates and real yields fell."), "application/pdf")},
        )
        assert pdf.status_code == 201
        assert pdf.json()["evidence_documents"][0]["kind"] == "pdf"
        assert pdf.json()["fact_claims"][0]["locator"] == "page 1"

        image = client.post(
            "/api/v1/agent/research-cases", headers=HEADERS,
            data={"question": "Read chart"},
            files={"file": ("chart.png", _png(), "image/png")},
        )
        assert image.status_code == 201
        assert image.json()["status"] == "degraded"
        assert image.json()["evidence_documents"][0]["extraction_status"] == "abstain"


def test_url_input_uses_bounded_fetch_and_creates_located_facts(monkeypatch):
    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body>ETF inflows confirmed the gold trend.</body></html>",
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with _client(monkeypatch, http=http) as client:
        result = client.post(
            "/api/v1/agent/research-cases", headers=HEADERS,
            data={"question": "gold flows", "url": "https://93.184.216.34/report"},
        )
        assert result.status_code == 201
        assert result.json()["evidence_documents"][0]["kind"] == "url"
        assert result.json()["fact_claims"]


def test_personalize_returns_agent_rules_api_without_persisting_profile(monkeypatch):
    with _client(monkeypatch) as client:
        case = client.post(
            "/api/v1/agent/research-cases", headers=HEADERS,
            json={"question": "gold outlook"},
        ).json()
        result = client.post(
            f"/api/v1/agent/research-cases/{case['case_id']}/personalize",
            headers=HEADERS,
            json={
                "risk_tolerance": "balanced", "horizon": "mid",
                "current_gold_pct": 18, "experience": "novice",
            },
        )
        assert result.status_code == 200
        payload = result.json()
        assert {"agent", "rules", "api", "watchlist", "invalidation", "next_review_at"} <= payload.keys()
        combined = str(payload)
        assert "建议买入" not in combined and "立即卖出" not in combined
        stored = client.get(
            f"/api/v1/agent/research-cases/{case['case_id']}", headers=HEADERS
        ).json()
        assert stored["investor_profile"] is None
        assert stored["personalized_brief"]["case_id"] == case["case_id"]


def test_missing_case_is_404(monkeypatch):
    with _client(monkeypatch) as client:
        assert client.get("/api/v1/agent/research-cases/rc_missing", headers=HEADERS).status_code == 404
