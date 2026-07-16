# P1: 大盘观点书 + 每周信号台账 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已验证的量化块组装成统一的短/中/长「大盘观点书」,并建立每周不可变信号台账(发布 + 影子组合计分),经网关端点暴露。

**Architecture:** `market_view.py` 纯函数式地从 `research_context` 的缓存 dict 组装观点书(不重算任何数字);`signal_ledger.py` 提供 append-only JSONL 台账(LedgerStore seam)+ 幂等每周发布 + 成熟周影子组合计分;`agent_gateway.create_app` 挂 5 个新只读端点。

**Tech Stack:** Python 3.11, pandas/numpy, FastAPI, pytest。无新依赖。

## Global Constraints

- LLM 不产生数字;本期无 LLM 参与,全部确定性。
- 台账 append-only + sha256 content hash;同 ISO 周发布幂等;历史记录不可重写、不回填。
- 一切块缺失时显式降级(沿用 research_context 的 degraded 语义)。
- 服务测试注入内存持久化,不依赖本机 Redis/Postgres/网络。
- 测试基线保持全绿(当前 204 passed)。

---

### Task 1: market_view.py — 观点书组装

**Files:**
- Create: `market_view.py`
- Test: `tests/test_market_view.py`

**Interfaces:**
- Consumes: `research_context.LocalQuantContext.get_context()` 的 dict(键:
  `vol_bands{h1,h5,h21}`, `regime_posterior.latest{calm,elevated,stress}`,
  `macro_factors{composite,factors_used,latest}`, `fair_value{deviation_pct,deviation_z,interpretation}`,
  `scenario_cone{checkpoints}`, `flagship{metrics}`, `allocation`, `data_asof`, `data_age_days`,
  `data_stale`, `is_realtime`, `degraded`)
- Produces: `build_market_view(ctx: Dict) -> Dict`,顶层键
  `{"short_term", "mid_term", "long_term", "meta"}`;每个 horizon 节:
  `{"available": bool, "core_view": str, "confidence": "低|中|高", "evidence": [str],
    "invalidation": [str], "data": {…原始数字引用…}}`;不可用时
  `{"available": False, "degraded_reason": str}`。`meta` 透传
  `data_asof/data_age_days/data_stale/is_realtime/degraded`。

核心规则(全部确定性、可测):
- short_term: 需要 `vol_bands`。core_view 用 h21 带宽描述区间与风险;波动带宽
  (p90-p10)/spot > 8% → confidence 低,否则中。invalidation: "已实现波动突破带宽"。
- mid_term: 需要 `regime_posterior` + `macro_factors`。主导状态 = argmax(latest);
  P(主导) ≥ 0.6 → 置信中,≥0.75 → 高,否则低。core_view 描述主导状态 + 因子复合方向
  (composite ≥0.55 顺风 / ≤0.45 逆风 / 中性)。invalidation: "P(主导状态) 跌破 0.5"、
  "因子复合翻越 0.5"。
- long_term: 需要 `fair_value`。用 deviation_z 描述估值(|z|<2 正常带内 / ≥2 结构性偏离),
  cone checkpoints d90 的 p10/p50/p90 作为情景区间;flagship 存在时附当前策略目标敞口语境。
  invalidation: "偏离 z 分突破 ±2"、"宏观锚回归关系失效(regime_break)"。

**Steps:**

- [ ] 1. 写失败测试 `tests/test_market_view.py`:
  - `test_full_context_builds_three_horizons`: 手工构造含全部块的 ctx,断言三节 available,
    每节有 core_view/confidence/evidence/invalidation 非空,meta 透传 data_asof。
  - `test_missing_blocks_degrade_explicitly`: ctx 只含 degraded 标记 →
    每节 available=False 且 degraded_reason 非空。
  - `test_confidence_rules`: P(stress)=0.8 的 posterior → mid_term confidence="高" 且
    core_view 提到压力状态;宽波动带 → short_term confidence="低"。
- [ ] 2. `pytest tests/test_market_view.py -v` → FAIL (module 不存在)
- [ ] 3. 实现 `market_view.py`(纯函数,无 IO)
- [ ] 4. 测试通过
- [ ] 5. commit `feat(P1): market_view 观点书组装器`

### Task 2: signal_ledger.py — 不可变台账 + 幂等发布

**Files:**
- Create: `signal_ledger.py`
- Test: `tests/test_signal_ledger.py`

**Interfaces:**
- Consumes: `market_view.build_market_view`,`research_context` ctx dict(allocation 块)。
- Produces:
  - `class LedgerStore(Protocol)`: `append(record: Dict) -> None`, `load_all() -> List[Dict]`
  - `class JsonlLedgerStore(LedgerStore)`: `__init__(path: str | Path)`
  - `class MemoryLedgerStore(LedgerStore)`(测试用)
  - `publication_id_for(date) -> str`(`"YYYY-Www"` ISO 周)
  - `build_publication(ctx: Dict, *, now: datetime) -> Dict`(含 content_hash)
  - `publish_weekly(store, ctx, *, now) -> Tuple[Dict, bool]`(record, created);
    同周已存在 → 返回已有记录 + created=False,**绝不覆盖**。

记录 schema:
```json
{"publication_id": "2026-W29", "published_at": "...UTC ISO", "data_asof": "2026-07-14",
 "data_source": "extended", "allocations": {"conservative": {"range_pct": [lo,hi], "midpoint": m}, ...},
 "market_view_summary": {"short_term": "...", "mid_term": "...", "long_term": "..."},
 "evidence_snapshot": {"regime_posterior": {...}, "fair_value_deviation_pct": x, "macro_composite": y},
 "degraded": {...}, "disclaimer": ALLOCATION_DISCLAIMER, "content_hash": "sha256:..."}
```
content_hash = sha256(除 hash 外字段的 canonical JSON)。

**Steps:**

- [ ] 1. 失败测试:
  - `test_publication_id_iso_week`(周一/周日边界)
  - `test_build_publication_has_hash_and_disclaimer`
  - `test_publish_weekly_idempotent`: 同周两次 → 第二次 created=False 且记录逐字节相同
  - `test_jsonl_store_append_only`(tmp_path): 两周发布 → 文件两行,重发布不改文件
  - `test_hash_detects_tampering`: 改字段后 verify 失败
- [ ] 2. RED → 3. 实现 → 4. GREEN → 5. commit `feat(P1): 不可变信号台账(幂等每周发布)`

### Task 3: 影子组合计分 score_track_record

**Files:**
- Modify: `signal_ledger.py`
- Test: `tests/test_signal_ledger.py`(追加)

**Interfaces:**
- Produces: `score_track_record(publications: List[Dict], prices: pd.Series, *, cost_bps: float = 5.0) -> Dict`
  返回 `{"per_profile": {profile: {"weeks_scored", "cum_return", "ann_vol", "max_drawdown", "sharpe"}},
  "benchmarks": {"static_midpoint": {...}, "gold_buy_hold": {...}}, "matured_through": str|None,
  "cost_bps": float, "disclaimer": str}`。

规则: 每个发布日取 `data_asof` 之后首个收盘价为调仓价;持仓 = midpoint% 金 + 其余现金(0 收益);
换手成本 = |Δ权重| × cost_bps;**只对存在下一发布(或样本末)且价格已成熟的周计分**;
无成熟周 → weeks_scored=0,指标为 None(诚实空账,不回填)。
基准: static_midpoint = 首次发布的 balanced 中点恒定持有;gold_buy_hold = 100% 金。

**Steps:**

- [ ] 1. 失败测试(合成价格序列 + 手算预期):
  - `test_empty_ledger_scores_empty`
  - `test_single_immature_week_not_scored`
  - `test_two_weeks_scored_with_cost`(手算 cum_return 断言到 1e-9)
  - `test_benchmark_gold_buy_hold_matches_price_ratio`
- [ ] 2. RED → 3. 实现 → 4. GREEN → 5. commit `feat(P1): 影子组合前向计分(成熟周,含成本)`

### Task 4: 网关端点

**Files:**
- Modify: `agent_gateway.py`(create_app 内,research_current 端点旁)
- Test: `tests/test_market_view_endpoints.py`

**Interfaces:**
- Consumes: Task 1-3 全部;`quant_research_context.get_context()`;
  app.state 注入 `signal_ledger_store`(默认 `JsonlLedgerStore("data_cache/signal_ledger.jsonl")`,
  create_app 参数可注入 MemoryLedgerStore)。
- Produces 端点(全部 public key + rate limit,模式复刻 research_current):
  - `GET /api/v1/agent/market-view` → build_market_view(ctx)
  - `GET /api/v1/signals/current` → 台账最新记录(空 → 404 `no_publication`)
  - `GET /api/v1/signals/history?limit=52`
  - `GET /api/v1/signals/track-record` → score_track_record(用 load_market_data 的金价)
  - `GET /api/v1/signals/{publication_id}` → 单条(404 `publication_not_found`)
  - `POST /api/v1/signals/publish`(**internal key only**)→ publish_weekly,返回 record+created

**Steps:**

- [ ] 1. 失败测试(TestClient + create_app(signal_ledger_store=MemoryLedgerStore())):
  auth 401/无 key;market-view 三节契约;publish(internal)→ current/history/{id} 读回;
  publish 幂等(两次 → created False);track-record 契约(空账 weeks_scored=0);
  publish 用 public key → 403。
- [ ] 2. RED → 3. 实现 → 4. GREEN(并跑全量 pytest 防回归)→ 5. commit `feat(P1): 观点书与信号台账端点`

### Task 5: 发布调度 + CLI

**Files:**
- Create: `scripts/publish_signal.py`
- Modify: `agent_gateway.py`(启动预热线程处,`SIGNAL_AUTOPUBLISH_ENABLED` 环境开关,默认关)
- Test: `tests/test_publish_signal.py`

**Interfaces:**
- Produces: `scripts/publish_signal.py::publish_once(store=None, now=None) -> dict`
  (CLI: `python3 scripts/publish_signal.py`,退出码 0/1);网关启动时若开关开且当日为周一
  且本周未发布 → 后台线程发布一次(失败只 log,不阻塞启动)。

**Steps:**

- [ ] 1. 失败测试: `test_publish_once_creates_record`(MemoryLedgerStore + 注入 ctx),
  `test_publish_once_idempotent_same_week`,`test_cli_main_returns_zero`。
- [ ] 2. RED → 3. 实现 → 4. GREEN → 5. commit `feat(P1): 信号发布 CLI 与启动自动发布开关`

### Task 6: 全量回归 + 文档

- [ ] 1. `python3 -m pytest -q` 全绿(204 + 新增)
- [ ] 2. README: 新端点加入 2.5 API 契约表;台账机制一段(不可变/幂等/前向计分/免责)
- [ ] 3. `.gitignore` 确认 `data_cache/` 已忽略(台账运行时文件不入库)
- [ ] 4. commit `docs(P1): API 契约与台账说明`
