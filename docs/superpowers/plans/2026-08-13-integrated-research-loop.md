# GoldenSense Integrated Research Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one auditable ResearchCase that connects evidence ingestion, eight-layer gating, expert research, three-horizon strategy, three-dimensional personalization, the existing four frontend workbenches, and forward accountability.

**Architecture:** Add three focused Python modules: a domain/store module, an Evidence Shield ingestion module, and a deterministic orchestrator. Mount thin FastAPI handlers in the existing gateway and reuse `research_context`, `market_view`, `personal_research`, and the signal ledger as authoritative computations. Share the active case across the existing React pages through a small local store and reusable case components.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, httpx, pypdf/pdfplumber/Pillow where available, React 18, Vite, Playwright, pytest.

---

### Task 1: ResearchCase domain and store

**Files:**
- Create: `research_case.py`
- Create: `tests/test_research_case.py`

- [ ] Write failing tests for strict `FactClaim`, `GateDecision`, `AgentView`, and `ResearchCase` schemas; accepted fact references; JSON-safe round-trip; and bounded in-memory store lookup.
- [ ] Run `python3 -m pytest tests/test_research_case.py -q` and confirm import failure.
- [ ] Implement Pydantic models with `extra="forbid"`, UUID case IDs, ISO timestamps, explicit status enums, and a thread-safe `ResearchCaseStore` with injected clock and `max_items` eviction.
- [ ] Re-run the focused test and commit.

Core interface:

```python
class ResearchCase(BaseModel):
    case_id: str
    question: str
    asset: Literal["XAUUSD"] = "XAUUSD"
    created_at: datetime
    data_asof: str | None
    status: CaseStatus
    evidence_documents: list[EvidenceDocument]
    fact_claims: list[FactClaim]
    gate_report: list[GateDecision]
    quant_pack: dict[str, Any]
    agent_views: list[AgentView]
    conflicts: list[ResearchConflict]
    horizon_strategy: dict[str, HorizonStrategy]
    personalized_brief: dict[str, Any] | None
    audit_report: AuditReport
    outcome_schedule: list[OutcomeCheckpoint]
```

### Task 2: Evidence Shield and URL/PDF/image ingestion

**Files:**
- Create: `evidence_shield.py`
- Create: `tests/test_evidence_shield.py`
- Modify: `requirements.txt`

- [ ] Write failing tests for HTTPS-only URLs, private/reserved DNS rejection, redirect revalidation, MIME/signature mismatch, 20MB PDF and 10MB image limits, PDF page citations, injection carriers, stale/replay signals, unit conflicts, and image OCR abstention.
- [ ] Run the focused test and confirm the expected failures.
- [ ] Implement `validate_public_https_url`, bounded fetch with redirect hook, PDF page extraction, optional local OCR adapter, SHA-256 evidence identity, fact extraction, and all eight gate decisions.
- [ ] Ensure untrusted text is converted into claims and never returned as prompt instructions. A blocking gate yields no accepted facts.
- [ ] Re-run tests and commit.

Core interface:

```python
async def ingest_evidence(
    *, question: str, url: str | None = None,
    filename: str | None = None, content_type: str | None = None,
    payload: bytes | None = None, http: httpx.AsyncClient | None = None,
) -> EvidencePacket: ...
```

### Task 3: Expert orchestration, model governance, and strategy arbitration

**Files:**
- Create: `research_orchestrator.py`
- Create: `tests/test_research_orchestrator.py`

- [ ] Write failing tests proving dynamic expert selection, accepted-fact-only evidence references, explicit minority conflicts, short-term abstention from unsupported direction, three horizon scenarios, and challenger models unable to override hard gates.
- [ ] Run the focused test and confirm import/behavior failures.
- [ ] Implement deterministic expert builders for macro/event, technical/flows, long-term fundamentals, and quant/model risk.
- [ ] Build a model registry that maps validated existing models to `production` and unvalidated ML/DL challengers to `watch`; expose OOS evidence and degradation reasons.
- [ ] Implement arbitration weighted by gate confidence, data freshness, horizon relevance, and available calibration; preserve disagreement in `conflicts`.
- [ ] Re-run tests and commit.

Core interface:

```python
def build_research_case(
    question: str, evidence: EvidencePacket,
    quant_ctx: dict[str, Any], *, mode: Literal["draft", "full"] = "full",
) -> ResearchCase: ...
```

### Task 4: FastAPI contracts and three-dimensional personalization

**Files:**
- Modify: `agent_gateway.py`
- Create: `tests/test_research_case_endpoints.py`
- Modify: `personal_research.py`

- [ ] Write failing endpoint tests for JSON question-only and multipart URL/PDF/image creation, authentication, limits, GET lookup, 404, and personalization.
- [ ] Add `ResearchCaseStore` to app lifespan and expose `POST/GET /api/v1/agent/research-cases` plus `POST /{case_id}/personalize`.
- [ ] Extend personalization with `agent`, `rules`, and `api` sections. Re-check market freshness/model state at request time and run directive-language audit before persisting the brief.
- [ ] Keep existing endpoints backward compatible and add optional `research_case_id` to the existing personal-research body without persisting the profile.
- [ ] Re-run focused and existing gateway/personalization tests and commit.

Expected personalization shape:

```json
{
  "case_id": "rc_...",
  "agent": {"core_conclusion": "...", "scenario_focus": []},
  "rules": {"risk_flags": [], "hard_constraints": []},
  "api": {"data_asof": "...", "freshness": "stale", "model_states": []},
  "watchlist": [], "invalidation": [], "next_review_at": "...",
  "disclaimer": "..."
}
```

### Task 5: Existing frontend integration

**Files:**
- Create: `modern_showcase_site/src/researchCaseStore.js`
- Create: `modern_showcase_site/src/ResearchCasePanel.jsx`
- Modify: `modern_showcase_site/src/App.jsx`
- Modify: `modern_showcase_site/src/QuantPage.jsx`
- Modify: `modern_showcase_site/src/SignalsPage.jsx`
- Modify: `modern_showcase_site/src/AdvisorPage.jsx`
- Modify: `modern_showcase_site/src/index.css`
- Modify: `modern_showcase_site/tests/e2e/goldensense.spec.js`

- [ ] Add a failing Playwright test for URL/PDF/image tabs, active case persistence, Evidence Shield expansion, shared case ID on all four routes, model governance, and the three personalization dimensions.
- [ ] Create a localStorage-backed active-case store and reusable case ribbon/audit components.
- [ ] Add the research intake panel to the existing dashboard without replacing current cards; use FormData and display upload/OCR degradation honestly.
- [ ] Add model governance to `/quant`, case strategy/change context to `/signals`, and case-linked three-dimensional output to `/advisor`.
- [ ] Preserve simple/pro progressive disclosure while never hiding stale/security/risk alerts.
- [ ] Run `npm run build` and Playwright, then commit.

### Task 6: Red-team, full-stack, and release verification

**Files:**
- Create: `eval/research_case_redteam.json`
- Create: `scripts/demo_research_case.py`
- Modify: `README.md`
- Modify: `DEPLOYMENT_DOC.md`

- [ ] Add clean/attacked pairs for prompt injection, table-unit mutation, stale replay, metadata carrier, and tool-return contamination.
- [ ] Assert clean cases remain parseable while attacked cases block, abstain, or lower confidence without changing a production strategy through untrusted claims.
- [ ] Add a three-minute deterministic demo script covering clean evidence, attacked evidence, strategy, personalization, and audit.
- [ ] Run focused tests, full `python3 -m pytest -q`, frontend build, Playwright, and a real local gateway smoke path.
- [ ] Document optional OCR dependencies, privacy lifecycle, API examples, limitations, and competition demo order; commit.

## Completion criteria

- All existing and new tests pass with no regression.
- URL, PDF, and image each produce a real ResearchCase or an explicit safe abstention.
- Four frontend workbenches display the same active case ID and data timestamp.
- Every agent evidence reference resolves to an accepted fact.
- Personalization exposes Agent, rules, and API sections and contains no directive trading language.
- No uploaded raw document or investor profile is persisted server-side.
