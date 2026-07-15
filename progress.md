# Progress Log

## Session 2026-07-15 — 全面落地策略与 Agent 架构改进 + 前端

### 最终状态：全部完成
- pytest: **166 passed / 0 failed**（会话开始时基线为 100 passed / 3 failed）
- 前端 `npm run build` 通过；/quant 页与 ⌘K 全局搜索经浏览器实测

### 新增模块（均带测试）
| 模块 | 内容 |
| --- | --- |
| data_sources.py | FRED(DFII10/T10YIE/DGS2) + yfinance 2004 起长历史 + GLD 成交额代理；缓存 data_cache/；产出 raw_market_data_extended.csv (5684 行) |
| vol_models.py | HAR-RV + P10/50/90 经验分位带 + 覆盖率诊断 |
| regime_probabilistic.py | numpy Gaussian HMM(EM, xi 向量化)，calm/elevated/stress 后验 |
| strategy_macro.py | 四因子中期组合（实际利率/美元/通胀预期/资金流代理）+ 回测 outputs/macro_backtest.csv |
| fair_value.py | 误差修正公允价值锚（滚动 10 年窗口；当前 +60%, r²=0.26） |
| allocation.py | BL-lite 配置区间（tilt 上限 ±25%）+ HMM 蒙特卡洛情景锥 |
| validation.py | purged walk-forward + PSR/DSR |
| meta_labeling.py | triple-barrier + XGB 过滤；真实数据 AUC 0.51 → 如实报告不部署 |
| analyst_committee.py | 四分析师 + regime 加权融合 + 分歧分数 |
| narrative_critic.py | 数字落地校验，失败回退规则文案 |
| outcome_tracker.py | 结果回填 + Brier/命中率 + 有界反哺 |
| research_context.py | 本地量化上下文 TTL 缓存（~0.6s 构建） |

### 网关改动 (agent_gateway.py)
- evaluate_regime_v2（HMM 后验概率混合暴露，短历史回退旧逻辑）
- 委员会接入 analyze 主链路；分歧只驱动 narrator 模型路由（不接管 stance 门控——教训：接管会让全部场景变"高风险观望"）
- 叙事校验器仅在 LLM 实际改写 draft 时运行（identity 比较）
- trace evidence bundle 增加 regime + committee
- 新端点：GET /research/current、GET /calibration；TraceStore.load_recent_analyses

### 前端 (modern_showcase_site)
- src/QuantPage.jsx（路由 /quant）、src/GlobalSearch.jsx（⌘K）、src/searchIndex.js
- index.css 追加 ~700 行黑金主题样式
- 实测：HMM 概率堆叠图（当前压力 92%）、情景锥（T+30 $3,635~$4,467）、校准记分卡（读到本机 trace 库真实历史：Brier 0.185）

### 22 年样本回测关键数字（2bps 成本）
- buy_and_hold: Sharpe 0.639, MaxDD **-44.4%**（2021-2026 短样本给出的 1.3 是 regime 运气）
- volmgd_long_moreira_muir: Sharpe 0.699, MaxDD -26%
- macro_gated_vol_target_10pct: Sharpe 0.606, MaxDD **-14.4%**

### 修复的既有问题
- 3 个环境敏感测试（本机 Redis 缓存泄漏进失败场景）：注入 _MemoryPersistence

### 后续可做
- carry 因子（需期货期限结构数据）、CoT 持仓、事件日历工具
- LLM-as-judge 评测护栏进 CI；TFT/FinBERT 走同一 walk-forward 门槛
- 部署：Railway 需重新构建镜像并跑 python3 data_sources.py 生成扩展数据
