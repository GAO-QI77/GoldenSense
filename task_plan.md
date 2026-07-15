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
- Status: in_progress
- Goal: 修三个硬阻断，让改进在生产真实生效且不误导
- Steps:
  - [ ] 阻断1: 扩展 CSV 进生产（提交 + data_cache/ 入 gitignore + Docker 构建期可再生）
  - [ ] 阻断2: research_context 暴露 data_asof/age/stale；/quant 加真实陈旧横幅
  - [ ] 阻断3: fair_value 增加偏离 z 分与"结构性偏离期"框定，前端改措辞
  - [ ] 验证: pytest 全绿 + 前端 build + 浏览器实测陈旧横幅与公允价值措辞

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

## Decisions Made
| Decision | Reason |
|----------|--------|
| HMM 用 numpy 手写 EM 而非 hmmlearn | 避免新增重依赖，且规模小可测 |
| 所有外部数据模块必须有离线回退 | 项目既有"诚实降级"纪律；CI 无网 |
