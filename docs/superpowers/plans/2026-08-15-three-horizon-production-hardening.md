# GoldenSense Three-Horizon Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose only short-, mid-, and long-term research horizons while making the connected GoldenSense stack deterministic, safe, auditable, and objectively scoreable above 90 when every hard gate passes.

**Architecture:** A public horizon contract and legacy adapter separate product language from old inference keys. Market reads become cache-only and validated, mutation endpoints receive dedicated throttles, and every public narrative passes one investment-language policy. Evidence, provenance, ledger labels, and frontend recovery states remain explicit rather than silently falling back.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, pandas/yfinance, pytest, React 18, Vite, Node test runner, Playwright.

---

### Task 1: Public Three-Horizon Contract

**Files:**
- Create: `horizon_contracts.py`
- Create: `tests/test_horizon_contracts.py`
- Modify: `agent_gateway.py`
- Modify: `service_contracts.py`
- Modify: `modern_showcase_site/src/App.jsx`
- Modify: `modern_showcase_site/tests/e2e/goldensense.spec.js`

- [ ] **Step 1: Write the failing public-contract tests**

```python
def test_public_horizons_are_research_periods_only():
    assert PUBLIC_HORIZONS == ("short_term", "mid_term", "long_term")

def test_legacy_adapter_is_one_way():
    assert to_legacy_quant_horizon("short_term") == "T+1"
    assert to_legacy_quant_horizon("mid_term") == "T+7"
    assert to_legacy_quant_horizon("long_term") == "T+30"
    assert public_horizon_payload("T+1") == "short_term"
```

Add an endpoint assertion that `summary_card.horizon` and every `horizon_forecasts[].horizon` use only the three public keys, and scan serialized public responses for `24h`, `7d`, `30d`, and `T+`.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `pytest -q tests/test_horizon_contracts.py tests/test_agent_analyze.py::test_agent_analyze_contract_ok`

Expected: failure because `horizon_contracts.py` does not exist and the current response uses `24h/7d/30d`.

- [ ] **Step 3: Implement the contract and legacy adapter**

```python
PublicHorizon = Literal["short_term", "mid_term", "long_term"]
PUBLIC_HORIZONS = ("short_term", "mid_term", "long_term")
_LEGACY_QUANT = {
    "short_term": "T+1",
    "mid_term": "T+7",
    "long_term": "T+30",
}

def to_legacy_quant_horizon(value: PublicHorizon) -> str:
    return _LEGACY_QUANT[value]
```

Rename all public loops, thresholds, labels, and frontend mock fields. Keep old strings only in functions or constants named `legacy_*` and in internal inference-service tests.

- [ ] **Step 4: Verify GREEN and scan for public leakage**

Run: `pytest -q tests/test_horizon_contracts.py tests/test_agent_analyze.py tests/test_research_case.py tests/test_research_orchestrator.py`

Run: `rg -n '24h|7d|30d|T\+' modern_showcase_site/src agent_gateway.py service_contracts.py`

Expected: tests pass; matches are limited to explicitly named legacy adapter code or non-horizon calendar copy such as `T-24h`.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add horizon_contracts.py tests/test_horizon_contracts.py agent_gateway.py service_contracts.py modern_showcase_site/src/App.jsx modern_showcase_site/tests/e2e/goldensense.spec.js
git commit -m "feat: expose research horizons as short mid and long"
```

### Task 2: Deterministic Market Trust Chain

**Files:**
- Modify: `market_snapshot_service.py`
- Modify: `data_loader.py`
- Modify: `service_contracts.py`
- Modify: `agent_gateway.py`
- Modify: `tests/test_market_snapshot_service.py`
- Modify: `tests/test_agent_analyze.py`

- [ ] **Step 1: Write failing concurrency and instrument-identity tests**

Create tests proving that 100 concurrent `GET /dashboard/current` calls never call the market refresh endpoint, all return the same price, and `GC=F` is labeled `COMEX_GOLD_FUTURES` rather than XAUUSD. Add a refresh test where a loader returns 4380 once and 99.67 next; assert the second value does not replace the trusted snapshot and is marked `degraded` with `instrument_identity_violation` or `price_jump_quarantined`.

- [ ] **Step 2: Run and confirm RED**

Run: `pytest -q tests/test_market_snapshot_service.py tests/test_agent_analyze.py -k 'market or dashboard'`

Expected: current toolbox posts `/refresh`, futures are labeled XAUUSD, and invalid values overwrite the cache.

- [ ] **Step 3: Implement cache-only reads, single-flight refresh, and validation**

Add an `asyncio.Lock` to the market service, validate finite positive values and ticker-specific ranges, compare gold against the last trusted snapshot, and quarantine extreme changes. Change `HttpResearchToolbox.get_market_snapshot()` to a GET of `market_snapshot_url`; only the internal refresh endpoint and background loop may fetch yfinance.

Represent instrument identity explicitly:

```python
InstrumentSnapshot(
    symbol="GC=F",
    label="COMEX 黄金期货",
    instrument_type="futures",
    quote_currency="USD",
    unit="USD/troy_oz",
)
```

- [ ] **Step 4: Verify deterministic behavior**

Run: `pytest -q tests/test_market_snapshot_service.py tests/test_agent_analyze.py -k 'market or dashboard'`

Run the local 100-request concurrency probe and assert exactly one unique `(symbol, latest_price)` pair and zero `status=ok` identity violations.

- [ ] **Step 5: Commit only Task 2 files**

```bash
git add market_snapshot_service.py data_loader.py service_contracts.py agent_gateway.py tests/test_market_snapshot_service.py tests/test_agent_analyze.py
git commit -m "fix: make market snapshots deterministic and identity safe"
```

### Task 3: Endpoint-Specific Rate Limits and Unified Investment Policy

**Files:**
- Create: `investment_output_policy.py`
- Create: `tests/test_investment_output_policy.py`
- Modify: `agent_gateway.py`
- Modify: `tests/test_agent_analyze.py`
- Modify: `modern_showcase_site/src/App.jsx`
- Modify: `modern_showcase_site/src/SignalsPage.jsx`
- Modify: `modern_showcase_site/src/AdvisorPage.jsx`

- [ ] **Step 1: Write failing tests for independent buckets and forbidden language**

Test that 100 allowed GET requests do not consume the analyze bucket, analyze overflow returns status 429 with `Retry-After`, `retry_after_seconds`, and a Chinese message, and optional prompt-injection text adds `prompt_injection_detected` without changing units. Assert every public narrative rejects phrases matching `目标仓位|满仓|加仓|减仓|买入|卖出|\d+倍杠杆`.

- [ ] **Step 2: Run and confirm RED**

Run: `pytest -q tests/test_investment_output_policy.py tests/test_agent_analyze.py -k 'rate or injection or directive'`

Expected: one limiter currently covers all routes, error copy is English, and the legacy analyzer can emit target exposure language.

- [ ] **Step 3: Implement named limiters and a shared policy**

Use buckets named `read`, `analyze`, `research_case`, `upload`, and `personalize`. Return retry metadata from `SlidingWindowRateLimiter.check()`. Add deterministic policy functions that convert prohibited execution language into research-language equivalents such as “参考区间差距” and “等待复核条件”，then run the policy over legacy and ResearchCase outputs before serialization.

- [ ] **Step 4: Verify GREEN and frontend recovery copy**

Run: `pytest -q tests/test_investment_output_policy.py tests/test_agent_analyze.py tests/test_research_case_endpoints.py`

Run: `npm run test:unit` in `modern_showcase_site`.

Expected: independent buckets pass; frontend renders Chinese retry text and never exposes the raw backend message.

- [ ] **Step 5: Commit only Task 3 files**

```bash
git add investment_output_policy.py tests/test_investment_output_policy.py agent_gateway.py tests/test_agent_analyze.py modern_showcase_site/src/App.jsx modern_showcase_site/src/SignalsPage.jsx modern_showcase_site/src/AdvisorPage.jsx
git commit -m "fix: separate throttles and unify investment safety policy"
```

### Task 4: Evidence Usefulness and Primary-Source Integrity

**Files:**
- Modify: `evidence_shield.py`
- Modify: `research_orchestrator.py`
- Modify: `research_case.py`
- Modify: `news_provenance.py`
- Modify: `tests/test_research_case_redteam.py`
- Modify: `tests/test_research_orchestrator.py`
- Modify: `tests/test_news_provenance.py`

- [ ] **Step 1: Write failing benign-PDF and source-integrity tests**

Build a benign document containing repeated headings plus unique factual paragraphs. Assert only duplicate claim clusters become `review`, unique traceable facts remain `accepted`, and the output audit cannot say “accepted claims are traceable” when the accepted count is zero. Test that competition rules route to `product_rules_review`, while Federal Reserve, CFTC, CME, WGC, and central-bank domains classify as primary and FXStreet remains secondary.

- [ ] **Step 2: Run and confirm RED**

Run: `pytest -q tests/test_research_case_redteam.py tests/test_research_orchestrator.py tests/test_news_provenance.py`

Expected: repeated headings currently downgrade the whole claim set and non-market documents fail to receive a suitable specialist.

- [ ] **Step 3: Implement cluster-level deduplication and product-rule routing**

Deduplicate normalized claims by semantic key while retaining the highest-quality locator. Add `product_rules_review` to AgentView roles, and keep its output separate from gold direction. Make output-audit state `abstain` when there are zero accepted facts.

- [ ] **Step 4: Verify clean/attack pairs**

Run the focused tests and one real local PDF case. Expected: benign material produces accepted cited facts; malicious instruction carriers remain review/block; no document rule changes the gold strategy without market evidence.

- [ ] **Step 5: Commit only Task 4 files**

```bash
git add evidence_shield.py research_orchestrator.py research_case.py news_provenance.py tests/test_research_case_redteam.py tests/test_research_orchestrator.py tests/test_news_provenance.py
git commit -m "feat: preserve benign evidence and route product rules safely"
```

### Task 5: Ledger Truth Labels and Frontend Usability

**Files:**
- Modify: `signal_ledger.py`
- Modify: `agent_gateway.py`
- Modify: `modern_showcase_site/src/QuantPage.jsx`
- Modify: `modern_showcase_site/src/SignalsPage.jsx`
- Modify: `modern_showcase_site/src/DashboardExperience.jsx`
- Modify: `modern_showcase_site/src/index.css`
- Modify: `modern_showcase_site/tests/unit/presentationPolicy.test.js`
- Modify: `modern_showcase_site/tests/e2e/goldensense.spec.js`
- Modify: `tests/test_signal_ledger.py`

- [ ] **Step 1: Write failing truth-label and touch-target tests**

Require every calibration metric and publication to carry `evidence_class` in `backtest|simulated_forward|live_forward`. Assert an empty ledger returns structured `first_publication_required` guidance rather than a generic 404 body. In Playwright at 390px, assert all primary buttons, news links, subscription controls, and input-type tabs have bounding-box height at least 44px.

- [ ] **Step 2: Run and confirm RED**

Run: `pytest -q tests/test_signal_ledger.py`

Run: `npm run test:unit && npx playwright test -g 'mobile|ledger'` in `modern_showcase_site`.

Expected: calibration lacks evidence-class labels and several mobile controls are 17–41px high.

- [ ] **Step 3: Implement labels, first-publication guidance, and accessible hit areas**

Keep empty history honest. Display historical backtest, simulated-forward, and live-forward results in separate cards. Add minimum 44px block-level hit areas without changing the approved black-gold visual system.

- [ ] **Step 4: Verify GREEN**

Run the focused backend and frontend commands. Expected: every result is truth-labeled and mobile checks pass without horizontal overflow.

- [ ] **Step 5: Commit only Task 5 files**

```bash
git add signal_ledger.py agent_gateway.py modern_showcase_site/src/QuantPage.jsx modern_showcase_site/src/SignalsPage.jsx modern_showcase_site/src/DashboardExperience.jsx modern_showcase_site/src/index.css modern_showcase_site/tests/unit/presentationPolicy.test.js modern_showcase_site/tests/e2e/goldensense.spec.js tests/test_signal_ledger.py
git commit -m "feat: truth-label outcomes and harden frontend recovery UX"
```

### Task 6: Full-Stack Acceptance and Objective Score

**Files:**
- Create: `scripts/competition_acceptance.py`
- Create: `tests/test_competition_acceptance.py`
- Modify: `README.md`
- Modify: `DEPLOYMENT_DOC.md`

- [ ] **Step 1: Write the failing acceptance-score test**

The script must calculate the six approved score categories, cap the score at 89 when any hard gate fails, and emit JSON containing `score`, `hard_gates`, `category_scores`, `evidence`, and `remaining_risks`.

- [ ] **Step 2: Run and confirm RED**

Run: `pytest -q tests/test_competition_acceptance.py`

Expected: failure because the acceptance scorer does not exist.

- [ ] **Step 3: Implement the deterministic scorer and operational documentation**

The scorer consumes actual test reports and live probes; it must not award points from hard-coded success booleans. Document production API keys, CORS, provider identity, refresh scheduling, per-bucket limits, and signal publication operations.

- [ ] **Step 4: Run fresh full verification**

Run: `pytest -q`

Run: `npm run test:unit && npm run build && npx playwright test` in `modern_showcase_site`.

Run the live service health check, 100-request concurrency probe, one clean PDF case, one attack case, four-page desktop journey, and 390px mobile journey.

- [ ] **Step 5: Generate the final objective report**

Run: `python3 scripts/competition_acceptance.py --json`

Expected: score above 90 only if every hard gate is true. Otherwise report the real capped score and the blocking evidence.

- [ ] **Step 6: Commit only Task 6 files**

```bash
git add scripts/competition_acceptance.py tests/test_competition_acceptance.py README.md DEPLOYMENT_DOC.md
git commit -m "test: add evidence-based competition acceptance scoring"
```
