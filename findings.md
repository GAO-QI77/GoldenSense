# Findings

## 环境基线 (2026-07-14)
- Python 3.13.5 本地（仓库基线 3.12，兼容）；已装：numpy 2.2.6, pandas 2.3.2, sklearn 1.7.1, scipy, statsmodels 0.14.6, torch 2.10, xgboost 3.1.3, lightgbm 4.6, yfinance 1.1.0, fastapi, httpx
- 网络可用：FRED fredgraph.csv 返回 200（无需 API key）
- 测试基线：`python3 -m pytest -q` → **100 passed, 3 failed（既有失败，非本次改动）**：
  - test_market_snapshot_service.py::test_market_readiness_fails_without_snapshot_when_fallback_disabled
  - test_news_ingest_service.py::test_recent_news_refresh_uses_sample_fallback_when_upstream_fails
  - test_news_ingest_service.py::test_news_readiness_fails_without_payload_when_fallback_disabled

## 代码结构关键点
- 回测：backtest_engine.run_backtest(forward_returns, positions, cost_bps)；positions[t] 吃 t 日 forward return
- regime_strategy.evaluate_regime(prices, risk_profile, vol_state) 被 agent_gateway.py:1658 调用；PROFILE_EXPOSURE_CAP={cons:40,bal:70,aggr:100}，VOL_EXPOSURE_SCALE={calm:1,elev:.7,stress:.4}
- gateway 关键类：HttpResearchToolbox(工具), OpenAINarrator(叙事), AgentAnalysisService.analyze_internal(主编排), AgentTraceStore(trace+反馈, Postgres 或 dev 内存)
- raw_market_data.csv 列：Date,Gold,Silver,USD_Index,S&P500,VIX,Crude_Oil,10Y_Bond,2Y_Bond；2021-03-16 起
- outputs/advanced_backtest.csv：B&H Sharpe 1.305 / DD -20.4%；multi_trend_volmgd_banded Sharpe 1.087 / DD -6.8%（2bps）
- walkforward: T+1 方向 OOS acc 44.35%，always_in Sharpe -1.2 → 方向预测无边际（Stage A 结论）

## 前端
- modern_showcase_site: Vite4 + React18 + Tailwind3 + framer-motion + lucide + react-router6
- 单文件 src/App.jsx (1620 行) + src/index.css (1744 行)，无 components 目录
- env: VITE_AGENT_API_URL / VITE_AGENT_DASHBOARD_URL / VITE_AGENT_FEEDBACK_URL / VITE_AGENT_API_KEY
- 已有路由与"研究终端"式页面；主题已偏暗色，需强化黑金科技感 + 全局搜索 + 新量化面板

## 设计决定
- 新增只读研究端点 GET /api/v1/agent/research/current（regime 概率/因子/公允价值/情景锥/校准），
  不动 dashboard 既有契约，前端新页面直接消费
- FRED 经由 fredgraph.csv?id=SERIES 拉取；缓存到 data_cache/；离线时回退现有 CSV 并打降级标记
