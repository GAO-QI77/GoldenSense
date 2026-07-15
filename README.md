# GoldenSense

GoldenSense 是一个面向中文用户的黄金投资辅助 Agent。它不执行交易，也不伪装成“自动赚钱系统”；它的目标是把量化预测、市场快照、新闻事件、历史类比和风险约束压缩成一份可追溯、可降级、可审计的分析结果。

当前仓库只保留正式 Agent 主链路，不再包含旧的直播平台或历史 demo 分支。

## 产品速览

GoldenSense 当前是一款可演示 MVP，面向需要快速理解黄金市场风险的中文个人研究用户。它把分散的行情、宏观指标、新闻事件、量化预测、历史类比与用户风险画像整理为一条可追溯的研究路径。

| 维度 | 当前设计 |
| --- | --- |
| 核心问题 | 黄金研究信号分散、结论难追溯、风险提示与用户情况脱节 |
| 用户主流程 | 研究主页 → 提交问题与风险画像 → 三周期分析 → 证据卡 / 引用 / 失效条件 → 风险提示 → 反馈 |
| 产品边界 | 研究辅助，不执行交易，不承诺收益，不提供交易所级实时行情 |
| 可信机制 | 数据新鲜度检查、显式降级、模型状态标记、多源冲突提示、风险画像门控、内部 Trace |
| 当前状态 | 已完成可演示 MVP；尚未披露真实用户使用数据 |
| 下一步验证 | 计划开展 5 - 10 位目标用户访谈与可用性测试，重点验证任务完成率、结果可理解性和风险提示识别率 |

## 演示入口

- GitHub 仓库：<https://github.com/GAO-QI77/GoldenSense>
- 消费者前台（本地）：`http://localhost:4173`
- 内部 QA 面板（本地）：`http://localhost:8501`
- Agent 网关（本地）：`http://localhost:8020`

公开前端体验入口：[https://goldensense-public-site.vercel.app](https://goldensense-public-site.vercel.app)。公网 Agent Gateway 已部署到 Railway：[https://agent-gateway-production-fa79.up.railway.app](https://agent-gateway-production-fa79.up.railway.app)。未配置 DeepSeek key 时，网关会返回保守的规则结果并保留降级标记。

## 仓库治理

- 许可证：MIT，见 [`LICENSE`](LICENSE)。
- 安全报告与生产硬化建议：见 [`SECURITY.md`](SECURITY.md)。
- 贡献流程与本地检查：见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
- CI：GitHub Actions 会在 `main` 和 PR 上运行 Python 测试、前端构建和 Playwright e2e。

## 设计目标

- 面向教育型投顾场景，而不是自动交易执行。
- 结论必须附带证据、失效条件和风险提示。
- 当输入数据陈旧、工具失败、证据冲突或风险过高时，系统必须诚实降级，而不是伪装成“正常无结果”。
- 调试与审计能力必须和公网入口隔离，避免生产环境的信息泄露。

## 非目标

- 不提供真实下单、仓位管理或经纪商接入。
- 不承诺实时 tick 级市场数据或交易所级 SLA。
- 不把 LLM 作为事实源；模型只负责叙事增强与输出组织，证据来自工具层。

## 架构概览

```mermaid
flowchart LR
    UI["Consumer Web<br/>modern_showcase_site"] --> GW["agent_gateway.py<br/>:8020"]
    QA["Internal QA Dashboard<br/>frontend/dashboard.py"] --> GW
    GW --> INF["inference_service.py<br/>:8010"]
    GW --> MEM["memory_service.py<br/>:8012"]
    GW --> MKT["market_snapshot_service.py<br/>:8014"]
    GW --> NEWS["news_ingest_service.py<br/>:8016"]
    MEM --> PG["Postgres / pgvector"]
    MKT --> REDIS["Redis"]
    NEWS --> REDIS
    GW --> TRACE["Trace Store<br/>Postgres or bounded dev-memory fallback"]
```

## 服务拓扑

| 组件 | 端口 | 职责 | 关键入口 |
| --- | --- | --- | --- |
| `inference_service.py` | `8010` | 输出 `T+1 / T+7 / T+30` 预测、概率和解释特征 | `POST /api/v1/forecast` |
| `memory_service.py` | `8012` | 返回历史相似事件及其后验金价表现 | `POST /api/v1/memory/search` |
| `market_snapshot_service.py` | `8014` | 统一市场快照、技术状态、波动率与新鲜度信息 | `GET /api/v1/market/snapshot/latest` |
| `market_snapshot_service.py` | `8014` | 基本面、技术面、宏观政策、资金情绪四类指标契约 | `GET /api/v1/market/indicators/current` |
| `news_ingest_service.py` | `8016` | 最近新闻归一化、去噪与新鲜度标注 | `GET /api/v1/news/recent` |
| `agent_gateway.py` | `8020` | 编排各工具并输出正式 Agent 响应与首页 BFF | `POST /api/v1/agent/analyze` / `GET /api/v1/agent/dashboard/current` |
| `frontend/dashboard.py` | `8501` | 内部 QA / 运营面板 | 走内部 `trigger` 接口 |
| `modern_showcase_site/` | `4173` | 面向终端用户的消费者前台 | 调用正式 `analyze` / `feedback` |

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| [`agent_gateway.py`](agent_gateway.py) | 正式 Agent 网关、鉴权、限流、审计与输出编排 |
| [`data_sources.py`](data_sources.py) | 扩展数据层：FRED 实际利率/盈亏平衡通胀 + 2004 年起长历史行情，带缓存与显式降级 |
| [`vol_models.py`](vol_models.py) | 短期层：HAR-RV 波动率预测 + P10/P50/P90 经验分位收益带 |
| [`regime_probabilistic.py`](regime_probabilistic.py) | 三状态 Gaussian HMM（numpy EM），输出 calm/elevated/stress 后验概率 |
| [`strategy_macro.py`](strategy_macro.py) | 中期层：实际利率/美元/通胀预期/资金流代理四因子组合（先验参数、含成本回测） |
| [`fair_value.py`](fair_value.py) | 长期层：实际利率+美元误差修正公允价值锚（滚动十年窗口） |
| [`allocation.py`](allocation.py) | BL-lite 配置区间（观点只倾斜画像先验）+ HMM 蒙特卡洛情景锥 |
| [`validation.py`](validation.py) | Purged walk-forward（带 embargo）+ PSR / Deflated Sharpe Ratio |
| [`meta_labeling.py`](meta_labeling.py) | Triple-barrier 元标签 + XGBoost 信号过滤研究框架（当前 AUC≈0.51，如实不部署） |
| [`analyst_committee.py`](analyst_committee.py) | 确定性四分析师委员会（技术/宏观/资金流/新闻）+ regime 加权融合 + 分歧分数 |
| [`narrative_critic.py`](narrative_critic.py) | 叙事校验器：LLM 输出中的每个数字必须能在证据包中落地，否则回退规则文案 |
| [`outcome_tracker.py`](outcome_tracker.py) | 结果回填与校准：命中率 / Brier 分数 / 有界置信度反哺 |
| [`research_context.py`](research_context.py) | 本地长历史量化上下文（TTL 缓存），供网关与 `/research/current` 使用 |
| [`inference_service.py`](inference_service.py) | 量化预测服务 |
| [`market_snapshot_service.py`](market_snapshot_service.py) | 市场快照服务 |
| [`news_ingest_service.py`](news_ingest_service.py) | 新闻摄取服务 |
| [`memory_service.py`](memory_service.py) | 历史事件检索 API |
| [`memory_ingestion.py`](memory_ingestion.py) | 历史事件 embedding 构建与入库 |
| [`service_contracts.py`](service_contracts.py) | 服务间契约模型 |
| [`scripts/dev_stack.sh`](scripts/dev_stack.sh) | 本地 Python 服务栈启动脚本 |
| [`scripts/smoke_agent.py`](scripts/smoke_agent.py) | 端到端冒烟脚本 |
| [`docker-compose.yml`](docker-compose.yml) | 本地 Compose 编排 |
| [`tests/`](tests) | 正式测试集 |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | GitHub Actions CI |

## 运行前提

- Python `3.12`
- Node.js `20`
- Docker / Docker Compose（推荐用于完整联调）
- PostgreSQL `16+`，如需向量检索建议启用 `pgvector`
- Redis `7`

`Python 3.12` 是当前代码、Docker 和 CI 的正式基线。不要把本仓库视为 `Python 3.13` 已支持项目。

## 快速开始

### 方案 A：Docker Compose 启完整栈

这是最接近正式联调的方式：

```bash
docker compose up --build
```

默认会启动：

- `redis`
- `postgres`
- `inference`
- `memory`
- `market_ingest`
- `news_ingest`
- `agent_gateway`
- `frontend`
- `webapp`

访问地址：

- 消费者前台：`http://localhost:4173`
- 内部 QA 面板：`http://localhost:8501`
- Agent 网关：`http://localhost:8020`

### 方案 B：本地 Python 服务栈

适合后端快速联调。确保当前激活的是 Python 3.12 环境，然后安装依赖：

```bash
python -m pip install -r requirements.txt
```

启动：

```bash
zsh scripts/dev_stack.sh start
```

查看状态：

```bash
zsh scripts/dev_stack.sh status
```

停止：

```bash
zsh scripts/dev_stack.sh stop
```

这套脚本默认以“本地容错友好”模式启动：

- `market_snapshot_service.py` 开启 `MARKET_ALLOW_SYNTHETIC_FALLBACK=1`
- `news_ingest_service.py` 开启 `NEWS_ALLOW_SAMPLE_FALLBACK=1`
- 两个 ingest 服务默认关闭后台轮询任务，便于本地调试

这意味着你即使没有完整外部依赖，也能把主链路跑起来；但响应可能带有显式降级标记。

### 启动前端

消费者前台：

```bash
cd modern_showcase_site
npm install
npm run dev
```

内部 QA 面板：

```bash
python3 -m streamlit run frontend/dashboard.py
```

## 配置与环境变量

参考文件：[`.env.example`](.env.example)

### 最小必配

```bash
export AGENT_PUBLIC_API_KEYS=dev-public-key
export AGENT_INTERNAL_API_KEYS=dev-internal-key
```

权限模型：

- `public key`：允许访问正式前台入口 `analyze` 和 `feedback`
- `internal key`：额外允许访问 `traces` 和 `trigger`

### Agent Gateway

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_ENV` | `development` | 除 `development` 外都会强制要求显式配置非默认 public / internal keys |
| `AGENT_PUBLIC_API_KEYS` | `dev-public-key`（仅 dev） | 逗号分隔的对外 API key 列表 |
| `AGENT_INTERNAL_API_KEYS` | `dev-internal-key`（仅 dev） | 逗号分隔的内部 API key 列表 |
| `AGENT_ANALYZE_RATE_LIMIT_PER_MINUTE` | `60` | `analyze` 限流阈值 |
| `AGENT_ANALYZE_RATE_LIMIT_WINDOW_SECONDS` | `60` | `analyze` 限流窗口 |
| `AGENT_ALLOW_ORIGINS` | 本地前端域名列表 | CORS 白名单 |
| `AGENT_TOOL_TIMEOUT_SECONDS` | `35.0` | 单工具总超时；需覆盖推理服务冷启动首次行情抓取 |
| `AGENT_TOOL_CONNECT_TIMEOUT_SECONDS` | `1.5` | 单工具连接超时 |
| `AGENT_ALLOW_TRACE_MEMORY_FALLBACK` | dev 默认 `1`，prod 默认 `0` | Trace store 数据库故障时是否允许退回进程内存 |
| `AGENT_TRACE_MEMORY_TTL_SECONDS` | `3600` | dev 内存审计缓存 TTL |
| `AGENT_TRACE_MEMORY_MAX_ITEMS` | `200` | dev 内存审计缓存上限 |
| `VIX_CIRCUIT_BREAKER_THRESHOLD` | `30` | 风险熔断阈值 |
| `INFERENCE_MODEL_CHECKPOINTS_DIR_T1` | `model_checkpoints` | T+1 模型 checkpoint 目录 |
| `INFERENCE_MODEL_CHECKPOINTS_DIR_T7` | `model_checkpoints` | T+7 模型 checkpoint 目录；默认不再指向不存在的目录 |

### 下游服务地址

| 变量 | 默认值 |
| --- | --- |
| `FORECAST_URL` | `http://localhost:8010/api/v1/forecast` |
| `MEMORY_URL` | `http://localhost:8012/api/v1/memory/search` |
| `MARKET_SNAPSHOT_URL` | `http://localhost:8014/api/v1/market/snapshot/latest` |
| `MARKET_INDICATORS_URL` | `http://localhost:8014/api/v1/market/indicators/current` |
| `RECENT_NEWS_URL` | `http://localhost:8016/api/v1/news/recent` |

### 数据与回退

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/postgres` | Postgres 连接串 |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 连接串 |
| `MARKET_DATA_PROVIDER` | dev 默认 `yfinance`，非 dev 默认 `external_required` | 行情 provider；生产应接入真实供应商 |
| `NEWS_DATA_PROVIDER` | dev 默认 `rss`，非 dev 默认 `external_required` | 新闻 provider；生产应接入真实供应商 |
| `MARKET_ALLOW_SYNTHETIC_FALLBACK` | dev 默认 `1`，非 dev 默认 `0` | 行情失败时是否允许生成样本快照 |
| `NEWS_ALLOW_SAMPLE_FALLBACK` | dev 默认 `1`，非 dev 默认 `0` | 新闻失败时是否允许回退缓存或样本流 |
| `MARKET_START_BACKGROUND_TASK` | `0` | 本地调试默认关闭后台刷新 |
| `NEWS_START_BACKGROUND_TASK` | `0` | 本地调试默认关闭后台刷新 |
| `NEWS_FETCH_TIMEOUT_SECONDS` | `4.0` | 新闻抓取超时 |
| `NEWS_STALE_AFTER_SECONDS` | `300` | 新闻陈旧阈值 |
| `NEWS_STALE_CACHE_GRACE_SECONDS` | `1800` | 陈旧缓存可接受窗口 |
| `INFERENCE_ALLOW_SYNTHETIC_FALLBACK` | dev 默认 `1`，非 dev 默认 `0` | 量化预测无法拉取原始输入时是否退回启发式代理 |
| `MEMORY_START_BACKGROUND_LOAD` | `0` | 是否在后台加载 embedding 模型；不会阻塞服务启动 |

### LLM 叙事层

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `deepseek` | `deepseek` 或 `openai`；未配置可用 key 时使用规则回退 |
| `DEEPSEEK_API_KEY` | 空 | DeepSeek 可选增强 key |
| `OPENAI_API_KEY` | 空 | OpenAI 可选增强 key；仅在 `LLM_PROVIDER=openai` 时使用 |
| `LLM_TIMEOUT_SECONDS` | `12.0` | LLM 单次请求超时；失败或非法 JSON 最多重试一次 |
| `AGENT_DEFAULT_MODEL` | `deepseek-v4-flash` | 默认叙事模型 |
| `AGENT_COMPLEX_MODEL` | `deepseek-v4-pro` | 证据冲突或高风险场景叙事模型 |

LLM 只负责叙事增强和输出组织，不承担证据检索、风险熔断或事实存储职责。

首发默认使用 `deepseek-v4-flash`；证据冲突或高风险场景才切到 `deepseek-v4-pro`。不要新接入 `deepseek-chat` 或 `deepseek-reasoner`：DeepSeek 官方已标注这两个兼容别名将在 `2026-07-24 15:59 UTC` 弃用。

## 初始化历史记忆库

`memory_service.py` 只有在数据库中存在 `historical_events` 时，才会返回真实的历史类比结果。初始化方式：

```bash
python3 memory_ingestion.py \
  --database-url postgresql://postgres:postgres@localhost:5432/postgres
```

默认行为：

- 市场数据读取 [`raw_market_data.csv`](raw_market_data.csv)
- 事件文本读取 [`perception_layer/news_mock_data.jsonl`](perception_layer/news_mock_data.jsonl)
- Embedding 模型使用 `sentence-transformers/all-MiniLM-L6-v2`

如果数据库不可用或检索失败，`memory_service.py` 会显式返回 `status=unavailable` 或 `status=degraded`，不会再伪装成“空结果就是没有历史相似事件”。

## 量化研究层（短 / 中 / 长期分层）

策略层按期限分工，每层使用不同的方法论，全部经过含成本回测与走前验证；纪律不变：
任何新信号必须过 `backtest_engine` 的成本感知门槛才能进入 Agent 输出。

| 期限 | 方法 | 模块 |
| --- | --- | --- |
| 短期 (T+1~T+5) | 不预测方向（已被走前验证证伪），改为 HAR-RV 波动率预测 + 经验分位收益带 | `vol_models.py` |
| 中期 (数周~数月) | 三状态 HMM 概率状态机 + 实际利率/美元/通胀预期/资金流四因子组合，概率加权暴露 | `regime_probabilistic.py`, `strategy_macro.py`, `regime_strategy.evaluate_regime_v2` |
| 长期 (6 个月+) | 公允价值锚（误差修正）+ BL-lite 配置区间 + regime-switching 蒙特卡洛情景锥 | `fair_value.py`, `allocation.py` |

数据基线：`python3 data_sources.py` 会从 FRED（无需 key）与 yfinance 拉取 2004 年起的扩展数据集
（含 10Y TIPS 实际利率、盈亏平衡通胀、GLD 成交额代理）写入 `raw_market_data_extended.csv`；
离线时所有消费方自动回退仓库内置 CSV。22 年样本包含 2008/2011-2015/2022 等多个 regime——
买入持有在该样本的最大回撤为 -44%，这也是所有策略结论的诚实基准。

Agent 编排新增三道确定性机制（均无 LLM 参与）：
- **分析师委员会**（`analyst_committee.py`）：四个专业视角输出立场与置信度，按波动状态加权融合；
  分歧分数超阈值时路由到更强叙事模型并记入 trace。
- **叙事校验器**（`narrative_critic.py`）：LLM 改写后的每个数字必须能在证据包内落地，
  否则回退确定性草稿并打 `narrative_critic_reverted` 降级标记。
- **校准闭环**（`outcome_tracker.py`）：到期分析自动回填实际走势、计算命中率与 Brier 分数，
  以硬上限反哺委员会置信度，并通过 `/api/v1/agent/calibration` 公开。

## 模型与回退行为

### 量化预测

`inference_service.py` 优先加载 checkpoint 进行真实预测；当模型不可用、输入准备失败或市场数据无法正常取得时，会退回启发式代理结果，并在响应中把 `forecast_basis` 标成 `heuristic_proxy`。响应还包含 `model_status`、`model_loaded` 和 `model_checkpoint_path`，用于区分真实模型输出与代理预测。

这意味着：

- `T+1 / T+7` 优先来自模型
- `T+30` 当前用于中期参考，不应当被解读为独立训练的长期预测系统
- 服务在不满足条件时倾向于“保守可用”，而不是“强行自信”

### 市场与新闻

- `market_snapshot_service.py` 在 development 可输出 `synthetic_fallback`，非 development 默认不允许伪装为真实行情
- `news_ingest_service.py` 在 development 可回退缓存或样本新闻，非 development 默认必须显式配置数据源或返回不可用
- `market`、`news` 和 `memory` 的服务契约都带有 `status`、`degraded_reason`、`source_freshness_seconds`

### Demo 资产策略

仓库内保留的 `model_checkpoints/`、`raw_market_data.csv`、`prediction_results.csv`、`selected_features.json` 和 `shap_summary.png` 是轻量 demo / research assets，用于让本地推理、训练和文档示例可复现。不要把私有生产数据、真实客户数据或大体积模型直接提交到 Git；后续大模型或大数据版本应通过 GitHub Release artifact、对象存储或 Git LFS 管理。

### Agent 输出

`agent_gateway.py` 会把下游降级汇总进：

- `degradation_flags`
- `risk_banner`
- `tool_trace`

所以前端和审计层可以区分“真的没有证据”与“工具失败导致证据不可用”。

## 正式 API 契约

### 1. 分析入口

```http
POST /api/v1/agent/analyze
Content-Type: application/json
X-API-Key: <public-or-internal-key>
```

请求体：

```json
{
  "question": "今晚 CPI 超预期的话，黄金 24 小时怎么看？",
  "optional_news_text": "美国 CPI 同比高于预期，美元与收益率同步走高。",
  "risk_profile": "conservative",
  "horizon": "24h",
  "locale": "zh-CN"
}
```

字段说明：

- `question`：用户表达层输入
- `optional_news_text`：证据层优先输入；如提供，Agent 会优先用它驱动新闻和历史类比检索
- `risk_profile`：`conservative | balanced | aggressive`
- `horizon`：`24h | 7d | 30d`
- `locale`：当前固定 `zh-CN`
- `investor_profile`：可选完整问卷；包含风险容量、周期、经验、资金占比、最大回撤、已有持仓、流动性需求、杠杆态度和投资目标。该字段只影响风险适配和建议强度，不改写三周期预测基线。

核心响应字段：

| 字段 | 说明 |
| --- | --- |
| `analysis_id` | 本次分析唯一标识 |
| `summary_card` | 主结论卡，包括 `stance`、`action`、`confidence_band`、失效条件 |
| `horizon_forecasts` | 固定返回 `24h / 7d / 30d` 三张卡 |
| `recent_news` | 最近新闻条目，最多 6 条 |
| `evidence_cards` | 结构化证据卡 |
| `citations` | 引用与出处摘要 |
| `risk_banner` | 风险等级与提醒 |
| `degradation_flags` | 工具降级标识 |
| `follow_up_questions` | 建议后续追问 |
| `timing_ms` | 时延拆解 |

### 2. 首页研究 BFF

```http
GET /api/v1/agent/dashboard/current
X-API-Key: <public-or-internal-key>
```

返回首页所需的稳定三周期预测、四类指标、近端新闻、数据质量和指标引用。该接口是消费者前台 `/` 的唯一研究首页入口。

### 2.5 量化研究与校准（新增）

```http
GET /api/v1/agent/research/current
X-API-Key: <public-or-internal-key>
```

返回本地长历史数据集（2004 年起）驱动的量化研究上下文：HMM 状态后验（`regime_posterior`）、
宏观因子面板（`macro_factors`）、公允价值锚（`fair_value`）、HAR-RV 分布带（`vol_bands`）、
90 日蒙特卡洛情景锥（`scenario_cone`）与按画像的配置区间（`allocation`）。每个区块要么给出数据，
要么在 `degraded` 中显式标注原因。消费者前台 `/quant` 页面即由该端点驱动。

```http
GET /api/v1/agent/calibration
X-API-Key: <public-or-internal-key>
```

系统的公开记分卡：对已到期的历史分析回填实际金价走势，输出方向判断命中率、Brier 分数、
分立场/分置信度拆解与最近判定列表。校准结果以硬上限（±20%）反哺委员会置信度，
`weight_adjustment.basis` 说明依据。

### 3. 反馈入口

```http
POST /api/v1/agent/feedback
Content-Type: application/json
X-API-Key: <public-or-internal-key>
```

```json
{
  "analysis_id": "uuid-from-analyze",
  "rating": "helpful",
  "comment": "解释很清楚"
}
```

### 4. 调试 / 审计入口

```http
GET /api/v1/agent/traces/{analysis_id}
X-API-Key: <internal-key>
```

会返回：

- 原始请求
- 工具调用轨迹
- 证据包
- 最终响应
- 用户反馈

这是内部运维接口，不应暴露给匿名公网。

### 5. 内部 QA 入口

```http
POST /api/v1/agent/trigger
X-API-Key: <internal-key>
```

该入口只为内部 QA / 运维保留，不是正式前台契约。

## `curl` 示例

分析：

```bash
curl -X POST http://127.0.0.1:8020/api/v1/agent/analyze \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: dev-public-key' \
  -d '{
    "question": "如果今晚 CPI 高于预期，黄金 24 小时怎么看？",
    "risk_profile": "conservative",
    "horizon": "24h",
    "locale": "zh-CN"
  }'
```

反馈：

```bash
curl -X POST http://127.0.0.1:8020/api/v1/agent/feedback \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: dev-public-key' \
  -d '{
    "analysis_id": "replace-with-analysis-id",
    "rating": "helpful"
  }'
```

拉取 trace：

```bash
curl http://127.0.0.1:8020/api/v1/agent/traces/replace-with-analysis-id \
  -H 'X-API-Key: dev-internal-key'
```

## 测试与验证

正式核心测试集：

```bash
python3 -m pytest -q \
  tests/test_agent_analyze.py \
  tests/test_news_ingest_service.py \
  tests/test_memory_service.py \
  tests/test_inference_service.py \
  tests/test_market_snapshot_service.py \
  tests/test_impact_breakdown.py \
  tests/test_vix_data.py
```

端到端冒烟：

```bash
python3 scripts/smoke_agent.py
```

CI 当前包含三层保障：

- Python 3.12 下的正式测试集
- Node.js 20 下的消费者前台生产构建
- Playwright Chromium 下的桌面与移动端 e2e

Docker Compose 主链路冒烟保留为本地/手动验证，避免公开 CI 过度依赖外部行情、新闻和容器冷启动时长。

## 生产部署建议

- 把 `APP_ENV` 设为 `production`
- 显式配置非默认的 `AGENT_PUBLIC_API_KEYS` 和 `AGENT_INTERNAL_API_KEYS`
- 显式配置生产级 `MARKET_DATA_PROVIDER` 与 `NEWS_DATA_PROVIDER`，不要依赖 dev provider
- 保持 `MARKET_ALLOW_SYNTHETIC_FALLBACK=0`、`NEWS_ALLOW_SAMPLE_FALLBACK=0`、`INFERENCE_ALLOW_SYNTHETIC_FALLBACK=0`
- 仅允许受信来源访问 `traces` 与 `trigger`
- 在反向代理或 API Gateway 层补充 TLS、IP 约束和速率限制
- 使用真实 Postgres / Redis，不依赖 dev 内存 trace fallback
- 将 LLM key 视为可选增强，而不是系统可用性的前提
- 部署后检查 `/health/live` 和 `/health/ready`；`ready` 失败时不应接入前端流量

## 公开 Demo 部署

黑客松公开入口采用轻量部署边界：

- Vercel 只部署 [`modern_showcase_site/`](modern_showcase_site/) 静态前台。
- Railway 使用 [`scripts/public_stack.py`](scripts/public_stack.py) 在一个容器中启动 5 个 Python 服务；只有 Agent Gateway 暴露公网端口。
- Neon 提供 Postgres / pgvector，Upstash 提供 Redis。
- `APP_ENV=demo` 显式启用 `yfinance + RSS` 和可观察降级；这是评委体验模式，不是交易级行情 SLA。

Railway、Neon 初始化和 Vercel 环境变量详见 [`DEPLOYMENT_DOC.md`](DEPLOYMENT_DOC.md)。

黑客松首发不购买行情或新闻 API。若公开体验需要更稳定的黄金现货价格，优先评估 Metals.Dev Silver：`$9.99/月`、`10,000 requests/月`，配合 5 分钟缓存使用。

## 已知限制

- 本项目是教育型辅助系统，不构成投资建议。
- `T+30` 仍是中期代理参考，不应当被解释为独立长期模型。
- 外部数据源主要用于研究与辅助判断，不是交易所级行情基础设施。
- `frontend/dashboard.py` 走内部入口，只适合 QA / 运营，不适合直接暴露给终端用户。

## 故障排查

### `memory_service` 总是返回 `unavailable`

优先检查：

1. `DATABASE_URL` 是否可连通
2. 是否已执行 `memory_ingestion.py`
3. Postgres 是否启用了 `pgvector`；没有也能运行，但会退回数组存储

### 分析结果里出现大量 `degradation_flags`

这通常意味着至少一个工具已降级。检查：

- `/health`
- 各服务端口是否正常
- `RECENT_NEWS_URL`、`MARKET_SNAPSHOT_URL`、`MEMORY_URL`、`FORECAST_URL` 是否配置正确
- 本地是否刻意开启了 fallback 模式

### 没有 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 会不会直接不可用

不会。系统会退回确定性规则生成，接口仍然可用，只是文案表达会更保守。

## 补充说明

- 所有 FastAPI 服务在直接启动时，默认都可通过 `/docs` 查看 OpenAPI 页面。
- 如果你只需要验证正式主链路，请优先使用 `analyze`、`feedback` 和 `traces`，不要把 `trigger` 当成对外契约。
