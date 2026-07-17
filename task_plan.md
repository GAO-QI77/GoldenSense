# Task Plan: 落地 GoldenSense 全部策略与 Agent 架构改进 + 金色科技感前端

## Goal
把上一轮评审中的全部建议落地为可运行、有测试的代码：
A) 量化策略层（短/中/长期分层）；B) Agent 架构升级（分析师委员会/分歧路由/校验器/校准闭环）；
C) 可搜索、美观、金色科技感的前端产品（modern_showcase_site 升级）。
纪律红线：全部新信号过成本感知回测；LLM 不碰数字；离线可运行（外部数据有降级回退）。

## Phases

### Phase 0: 环境勘察与基线
- Status: complete（基线 100 passed / 3 pre-existing failed，见 findings.md）

### Phase 1: 数据层 (P0)
- Status: complete
- 产物: data_sources.py + tests/test_data_sources.py (5 passed)
- 结果: raw_market_data_extended.csv 5684 行 2004-01-05→2026-07-14，
  含 Real_10Y/Breakeven_10Y/2Y_CMT/GLD_DollarVolume，全部源 OK

### Phase 2: 短期层 (P0) — 波动率与分布
- Status: complete — vol_models.py (HAR-RV + 分位带 + 覆盖率诊断), 6 tests passed

### Phase 3: 中期层 (P1) — 概率状态机 + 四因子
- Status: complete — regime_probabilistic.py (numpy HMM, xi 向量化), strategy_macro.py
  (4 因子: real_rate/usd/breakeven/flow_proxy), evaluate_regime_v2 概率混合;
  22 年回测: B&H Sharpe 0.639 / DD -44%; macro_gated_vol_target DD -14.4%
- 注: carry 因子需期货期限结构数据，已在 docstring 标注为 roadmap

### Phase 4: 长期层 (P2) — 公允价值 + 配置 + 情景锥
- Status: complete — fair_value.py (滚动10年窗口, dev +60%, r²=0.26),
  allocation.py (BL-lite ±25% tilt caps + HMM 蒙特卡洛锥), 11 tests passed

### Phase 5: 验证纪律升级 (P1)
- Status: complete — validation.py (purged WF + PSR/DSR), meta_labeling.py
  (triple-barrier + XGB 过滤; 真实数据 AUC 0.51 → 如实报告无边际, 不部署), 10 tests

### Phase 6: Agent 编排升级 (P1/P2)
- Status: complete — 模块 + gateway 接线全部完成；关键设计修正：
  委员会分歧只驱动模型路由与置信度，不接管 stance 门控（否则 mock 场景全变高风险观望）
- 端点: GET /api/v1/agent/research/current, GET /api/v1/agent/calibration
- tests/test_research_endpoints.py 5 passed；test_agent_analyze 全绿

### Phase 7: 前端 — 金色科技感可搜索研究终端
- Status: complete — /quant 页 (QuantPage.jsx): HMM 概率堆叠图/情景锥/公允价值/
  因子面板/波动带/配置区间/校准记分卡；GlobalSearch.jsx (⌘K 搜索覆盖页面/面板/新闻/模板);
  searchIndex.js pub/sub 索引；CSS 黑金主题扩展
- 浏览器实测: /quant 全面板渲染、⌘K 搜索→过滤→点击跳转 /agent 成功、控制台无错误
- 注意: 浏览器点击坐标 = 截图像素空间(800x450)，勿用 2 倍坐标

### Phase 8: 全量验证与文档
- Status: complete
- pytest: 166 passed / 0 failed（基线 3 个环境敏感失败已修：注入 _MemoryPersistence
  隔离本机 Redis 缓存）；npm run build 通过
- README 更新：仓库结构表、量化研究层章节、2.5 API 契约
- .claude/launch.json 新增 gateway/web 启动配置

### Phase 9: A 档上线阻断修复
- Status: complete（提交步骤因分类器临时不可用，末尾重试）
- Steps:
  - [x] 阻断1: data_cache/ 入 gitignore；扩展 CSV 暂存入提交；research_context
        检测扩展数据缺失时打 extended_dataset 降级标记
  - [x] 阻断2: research_context 暴露 data_asof/data_age_days/data_stale/is_realtime；
        QuantPage DataFreshnessBanner（实测"截至 2026-07-14 · 1 天前收盘 · 非实时行情"）
  - [x] 阻断3: fair_value 增 deviation_z/band_std_pct/regime_break/interpretation；
        前端按 z 分显示"结构性偏离期"或"正常估值波动"（实测 +60.1% → 1.5σ → 正常带内）
  - [x] 验证: pytest 170 passed（+4 新测试）；前端 build 通过；浏览器实测横幅+公允价值+无控制台错误

### Phase 10: B 档 — 真实产品上线（自动推进）
- Status: complete（代码项 B1-B4）
- 已落地（代码 + 测试）:
  - [x] B1 定时数据刷新: scripts/refresh_data.py（cron 入口 + 内嵌调度线程，
        DATA_REFRESH_ENABLED），public_stack 接入；失败保留 last-good CSV。5 tests
  - [x] B2 模型自动降级: model_governance.py（命中率<45%→demote 保守化，
        <50%→观察期，有界置信度），gateway TTL 缓存 + summary_card 覆盖 +
        model_demoted_by_performance 标记 + /calibration 暴露 governance。8+2 tests
  - [x] B3 端点监控: service_metrics.py（per-route 时延/错误率/p95 + 域计数器）
        + record_metrics 中间件 + 内部 /metrics 端点。4 tests
  - [x] B4 LLM 评测护栏: eval/（golden_set.jsonl + judges 确定性 + run_eval CLI +
        stub toolbox），tests/test_eval_harness.py 进 CI。硬门: 忠实度/风险/失效条件。8 tests
  - [x] 硬化收尾: HMM/公允价值数值路径 np.errstate 抑制退化告警（日志洁净）
  - [x] 前端: 校准面板 GovernanceBadge（冠军/观察/降级/样本不足）
- 结果: pytest 195 passed；前端 build 通过；/quant 治理徽章浏览器实测正常
- 用户决策已回复:
  - [x] B5 → 保持免费源 + 诚实标注: README 已知限制补显式声明；机制(非实时横幅/
        is_realtime=false/降级标记)已在 A 档落地，无需新数据源
  - [x] B6 → 加强产品内免责 + 去建议化: allocation 改"研究参考区间"+ 免责字段，
        前端标题/图例/免责横幅去建议化，regime reason 去指令化，README 补免责。
        recommended_range_pct → reference_range_pct（as_dict 保留旧键兼容）

### Phase 11: 旗舰策略（真实提升）+ 抓眼球前端
- Status: complete
- 真实结果(2004-2026, 2bps, 全因果): 旗舰 Sharpe 0.70 / Sortino 1.01 / 回撤 -21.0% /
  Calmar 0.24 / DSR 0.95  vs  买入持有 0.64 / 0.90 / -44.4% / 0.24
- 关键诚实修正: HMM 从平滑后验(含未来)改为 filter_posterior 前向滤波 +
  causal_regime_stress 走前重拟合(expanding window)，零前视
- 产物: strategy_integrated.py, regime_probabilistic.filter_posterior/causal_regime_stress,
  research_context.flagship, 网关启动预热线程, QuantPage FlagshipHero(回撤/净值切换+动效)
- 测试: 204 passed（+9）；前端 build 通过；浏览器实测回撤/净值双视图+动效+无控制台错误

### Phase 13: 双 Agent 投研系统（分支 codex/dual-agent-system，2026-07-16）
- 设计: docs/superpowers/specs/2026-07-16-dual-agent-research-system-design.md（已获批准）
- P1 观点书+信号台账: complete — market_view.py（三尺度+失效条件）、signal_ledger.py
  （不可变 JSONL/幂等/sha256/影子组合成熟周计分）、5+1 网关端点、publish CLI+自动发布开关
- P2 个性化 Agent: complete — investor_profile.py、personal_research.py（规则引擎+去指令化
  检查）、narrate_personal（DeepSeek）、personal-research 端点（critic+去指令化双硬门，
  画像不落库）、评测 4 硬门 judge+3 golden 用例。注意: 网关已有同名 InvestorProfile
  （analyze 问卷），导入用 PersonalResearchProfile 别名
- P3 RAG: complete — knowledge/events_catalog.jsonl（52 真实事件）、event_study.py
  （前向收益零手写，mp 类 30 天均值 +3.61% n=19）、event_classifier.py、news_archive.py
  （相关性过滤/去重/180 天 TTL/衰减检索/CJK 二元切分）、knowledge/search 端点
- P4 新闻反应: complete — news_analyst 类比先验（±0.3 硬上限，n<5 不启用，不碰 stance
  门控）、research_context.invalidate（15min 防抖）、高严重度事件触发
- P5 前端: complete — AdvisorPage/SignalsPage、首页观点书面板、vite 代理、浏览器实测通过
- P6 致命层: complete —
  ①统一画像体系: 画像进阶层(4 可选字段+3 错配规则)、前端共享 profileStore(v2+v1迁移)、
    AgentPage 弃 9 字段问卷改共享画像+toLegacyAnalyzeProfile 适配器(/analyze 契约零改动)、
    风险预算全%化去建议化；两页画像实测互通
  ②两段式生成: mode=draft 端点(实测 19ms 出全部数字)→ 前端草稿秒渲染+「润色中」状态
    → DeepSeek 完成后原位替换(浏览器实测完整闭环)
  ③事件横幅: /event-alert 端点(5min 进程缓存,降级不报错)+ EventAlertBanner 全站挂载
- 测试: 305 passed / 0 failed(64s)；评测门 8/8；e2e 8/8；npm build 通过
- DeepSeek: .env key 已配置并实测接通(generated_by=llm,critic 4 数字全着地)；
  LLM_TIMEOUT_SECONDS 上调 45s(12s 会超时)
- 重要修复: env_loader 引入后 .env 的 INFERENCE_MODEL_CHECKPOINTS_DIR_T1 覆盖测试
  显式参数导致套件死锁——已修优先级(显式>env)+ conftest 剥离 .env 影响
  (LLM key/预热线程/checkpoint 目录),测试套件与开发者 .env 完全隔离

### Phase 12: 推送、PR、CI、部署
- Status: blocked-on-user（仅剩合并一步）
- 已完成:
  - [x] push codex/goldensense-public-demo → origin（新远端分支，未动 main）
  - [x] 开 PR #2 → main（github.com/GAO-QI77/GoldenSense/pull/2）
  - [x] CI clean env 全绿: Python tests pass 5m54s, Web build+e2e pass 1m1s
  - [x] 评测门 5/5 passed
- 待用户:
  - [ ] 合并 PR #2（自我审批被安全策略挡下——需人来点合并，这是正确红线，不绕过）
  - [ ] 合并后确认 Vercel/Railway 部署；生产切换设 APP_ENV=production + 真 Postgres + 法务确认

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| gh pr merge 被 classifier 拒绝 | 1 | 正当红线(自我审批/双人审查)，不绕过，交用户合并 |

## Decisions Made
| Decision | Reason |
|----------|--------|
| HMM 用 numpy 手写 EM 而非 hmmlearn | 避免新增重依赖，且规模小可测 |
| 所有外部数据模块必须有离线回退 | 项目既有"诚实降级"纪律；CI 无网 |
