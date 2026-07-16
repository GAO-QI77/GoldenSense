# P2: 个性化研究 Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans。前端 /advisor 页统一放 P5。

**Goal:** 第二支柱 Agent:投资者画像 → 确定性规则引擎(全部数字)→ DeepSeek 定制叙事(critic + 去指令化双重把关)→ `POST /api/v1/agent/personal-research`。

**Architecture:** `investor_profile.py`(画像与叙事的 Pydantic 模型)→ `personal_research.py`(规则引擎: 参考区间/仓位差距/风险旗标/期限证据 + 确定性草稿叙事 + 去指令化检查器)→ 网关端点(narrator 扩展 `narrate_personal`,critic 复用 `verify_narrative`,双检不过全部回退草稿并打降级标)。服务端不持久化画像。

**Tech Stack:** 现有栈,无新依赖。

## Global Constraints
- LLM 不产生数字;所有数字来自 allocation/market_view/research_context。
- 输出框架 = 参考区间/差距/风险提示;禁止指令式措辞(检查器硬门)。
- 免责声明必须在响应中出现;画像不落库。
- 离线(无 DEEPSEEK_API_KEY)→ 确定性草稿,功能完整。

### Task 1: investor_profile.py 画像与叙事模型
- `InvestorProfile(BaseModel, extra=forbid)`: risk_tolerance ∈ {conservative,balanced,aggressive}; horizon ∈ {short,mid,long}; current_gold_pct: float [0,100]; experience ∈ {novice,experienced,professional}
- `PersonalNarrative(BaseModel)`: overview, position_analysis, risk_notes: List[str], horizon_note, disclaimer
- 测试: 合法通过/非法值 422 语义(ValidationError)/边界 0 与 100

### Task 2: personal_research.py 规则引擎
- `build_personal_facts(profile, ctx) -> Dict`:
  - reference_range: ctx["allocation"][risk_tolerance](缺失→显式降级)
  - position_gap: {"status": below|within|above, "gap_pct": float}(相对区间)
  - risk_flags: 结构化列表,规则:
    - `position_above_range_in_stress`: above 且 P(stress) ≥ 0.35
    - `position_far_above_range`: 超上沿 > 5 个百分点
    - `short_horizon_high_vol`: horizon=short 且 h21 带宽 > 8%
    - `structural_valuation_deviation`: |deviation_z| ≥ 2
    - `stale_data`: ctx.data_stale
  - horizon_evidence: 按 horizon 选观点书对应节(build_market_view 复用)
  - 每条含 evidence_ref 字段路径
- `draft_personal_narrative(facts, profile) -> PersonalNarrative`: 确定性中文模板,经验等级决定解释深度(novice 加术语解释)
- `check_no_directive_language(texts) -> Tuple[bool, List[str]]`: 禁词模式(你应该买/卖、建议买入/卖出、立即加/减仓、满仓、清仓、抄底、赶紧、必须买 等)
- 测试: 三种 gap 状态;各风险旗标触发/不触发;草稿含免责;检查器抓禁词/放行合规文本

### Task 3: 网关端点 + narrator 扩展
- `OpenAINarrator.narrate_personal(facts, profile, draft) -> PersonalNarrative`(deepseek/openai 双路,超时/空输出/解析失败回退草稿,模式复刻 narrate)
- `POST /api/v1/agent/personal-research`(public key + rate limit):
  facts → draft → narrate_personal → verify_narrative(数字着地) → check_no_directive_language
  → 任一不过: 回退草稿 + degradation_flags(narrative_critic_reverted / directive_language_reverted)
  → 响应 {profile_echo, facts, narrative, degradation_flags, generated_by}
- 测试: 契约(字段齐全/免责在场);注入 narrator 返回未着地数字→ 回退+标记;注入返回指令化措辞→ 回退+标记;无 key 环境→ generated_by=deterministic_draft
- 全量回归

### Task 4: 评测护栏
- eval/judges.py 增 `judge_no_directive_language`(复用检查器)
- eval/golden_set.jsonl 增 personal 用例(3 条画像;expect: 免责/无指令化/数字着地)
- run_eval 支持 personal 通道(离线 stub)
- 测试进 CI(tests/test_eval_harness.py 扩展)

### Task 5: README 契约 + 提交
