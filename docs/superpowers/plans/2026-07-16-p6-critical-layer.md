# P6: 致命层修复 Implementation Plan

> REQUIRED SUB-SKILL: superpowers:executing-plans

**Goal:** 修复 PM 评审致命层三项:①统一画像体系(双入口合并)②个性化生成异步化(1 秒出数字,LLM 后补语言)③重大事件首页横幅(名实落差防线)。

**Architecture:**
- ①后端 `InvestorProfile` 增 4 个**可选**进阶字段(向后兼容,旧 4 字段请求不变);规则引擎新增进阶规则(回撤承受力错配/杠杆态度/流动性-期限错配)。前端新建共享 `profileStore.js`(v2 schema + v1 迁移),AdvisorPage 加可折叠进阶区,AgentPage 弃 9 字段问卷改用共享画像 + **客户端适配器**映射到 /analyze 旧契约(后端 /analyze 零改动零测试破坏),文案去建议化(¥→组合 %,"建议"→"研究参考")。
- ②`POST /personal-research?mode=draft|full`(默认 full):draft 跳过 LLM 秒回。前端两段式:先 draft 渲染全部数字+骨架提示"DeepSeek 正在润色语言",full 返回后原位替换;失败保留草稿。
- ③`GET /api/v1/agent/event-alert`:toolbox.search_recent_news → event_classifier 逐条分类 → 返回近窗口内最高严重度事件(TTL 300s 进程缓存);前端 `EventAlertBanner` 挂 AppShell(高严重度才显示,醒目金红,链接 /signals)。

**Global Constraints:** /analyze 后端契约不动;/personal-research 向后兼容;全部新逻辑有测试;去建议化措辞红线;pytest 全绿 + e2e 全绿 + 浏览器实测。

### Task 1: 画像模型进阶字段 + 规则(后端,TDD)
- investor_profile.py: `max_drawdown_pct: Optional[float] (0-100)`, `liquidity_need: Optional[low|medium|high]`, `leverage_attitude: Optional[none|low|medium|high]`, `investment_goal: Optional[capital_preservation|income|event_trade|trend_following|speculation]`
- personal_research.py 新规则(仅当字段提供时):
  - `drawdown_tolerance_mismatch`: 仓位 × |21d 带 p10| > max_drawdown_pct/100 → 声明的回撤承受力可能被当前仓位在一个坏月内击穿
  - `leverage_out_of_scope`: leverage_attitude ∈ {medium,high} → 研究口径不含杠杆,杠杆放大区间外风险
  - `liquidity_horizon_mismatch`: liquidity_need=high 且 horizon=long → 流动性需求与期限矛盾
- 草稿叙事纳入进阶结论;测试:各规则触发/不触发/字段缺省不触发/旧 4 字段请求完全不变

### Task 2: mode=draft 异步支撑(后端,TDD)
- 端点加 query `mode: Literal["draft","full"]="full"`;draft → 不调 narrator,generated_by="deterministic_draft",响应加 `"mode":"draft"`
- 测试:draft 不触碰 narrator(注入 narrator 若被调用即 raise);full 行为不变

### Task 3: event-alert 端点(后端,TDD)
- `GET /api/v1/agent/event-alert`(public):toolbox.search_recent_news("黄金 宏观 政策 风险", limit=6) → classify_news(title+summary) → 取 severity=high 的最新一条;无 → `{"active": false}`;有 → `{"active": true, "severity", "category", "title", "published_at", "checked_at"}`;app.state 缓存 TTL 300s;toolbox 失败 → active:false + degraded 字段
- 测试:高严重度新闻场景 → active true;普通新闻 → false;toolbox 抛错 → false+degraded;TTL 缓存命中不二次调用

### Task 4: 前端共享画像 + AgentPage 改造
- `src/profileStore.js`: load/save/subscribe,key `gs_profile_v2`,自动迁移 v1(4 字段)与 AgentPage 旧 9 字段无迁移(直接默认);`toLegacyAnalyzeProfile(profile)` 适配器(映射表:conservative→low 等,current_gold_pct>0→long,进阶字段透传或默认)
- AdvisorPage: 用 profileStore;新增「进阶画像(可选)」折叠区(4 个进阶字段);两段式生成(draft→即时渲染+narrative 区 shimmer「DeepSeek 正在润色语言…」→full 替换;full 失败保留草稿+提示)
- AgentPage: 删 9 字段问卷 UI,改用共享画像(核心+进阶,同组件);提交时经适配器发旧契约;RiskBudgetPanel/SuitabilityGate 文案与计算去 ¥ 化(以组合 % 表达),「建议暴露上限」→「研究参考暴露上限」;页头注明「画像与个性化研究页共享」
- 复用 SegmentedRow 等组件抽到 `src/profileFields.jsx` 避免双份

### Task 5: EventAlertBanner(前端)
- `src/EventAlertBanner.jsx`: 挂 AppShell(全站),GET event-alert,active 时显示醒目横幅(金红渐变+脉冲圆点):「重大市场事件:{title} · 观点书将在下次刷新纳入影响」+ 链接 /signals;可手动关闭(sessionStorage 记忆);CSS 加 `.event-alert-banner`
- e2e mock 该端点避免既有用例受扰(检查现有 spec 是否需要加 route mock)

### Task 6: 全量验证
- pytest 全绿;npm build;playwright e2e 全绿;浏览器实测:advisor 两段式(草稿秒出→LLM 替换)、agent 页新画像、横幅(用 mock 高严重度验证);commit 分批
