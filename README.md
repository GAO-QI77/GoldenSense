# GoldenSense

GoldenSense 是一个面向中文用户的黄金投资辅助 Agent。它不执行交易，也不伪装成“自动赚钱系统”；它的目标是把量化预测、市场快照、新闻事件、历史类比和风险约束压缩成一份可追溯、可降级、可审计的分析结果。

当前仓库只保留正式 Agent 主链路，不再包含旧的直播平台或历史 demo 分支。

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
| `news_ingest_service.py` | `8016` | 最近新闻归一化、去噪与新鲜度标注 | `GET /api/v1/news/recent` |
| `agent_gateway.py` | `8020` | 编排各工具并输出正式 Agent 响应 | `POST /api/v1/agent/analyze` |
| `frontend/dashboard.py` | `8501` | 内部 QA / 运营面板 | 走内部 `trigger` 接口 |
| `modern_showcase_site/` | `4173` | 面向终端用户的消费者前台 | 调用正式 `analyze` / `feedback` |

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| [`agent_gateway.py`](agent_gateway.py) | 正式 Agent 网关、鉴权、限流、审计与输出编排 |
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
| `APP_ENV` | `development` | 生产环境会强制要求显式配置 public / internal keys |
| `AGENT_PUBLIC_API_KEYS` | `dev-public-key`（仅 dev） | 逗号分隔的对外 API key 列表 |
| `AGENT_INTERNAL_API_KEYS` | `dev-internal-key`（仅 dev） | 逗号分隔的内部 API key 列表 |
| `AGENT_ANALYZE_RATE_LIMIT_PER_MINUTE` | `60` | `analyze` 限流阈值 |
| `AGENT_ANALYZE_RATE_LIMIT_WINDOW_SECONDS` | `60` | `analyze` 限流窗口 |
| `AGENT_ALLOW_ORIGINS` | 本地前端域名列表 | CORS 白名单 |
| `AGENT_TOOL_TIMEOUT_SECONDS` | `4.0` | 单工具总超时 |
| `AGENT_TOOL_CONNECT_TIMEOUT_SECONDS` | `1.5` | 单工具连接超时 |
| `AGENT_ALLOW_TRACE_MEMORY_FALLBACK` | dev 默认 `1`，prod 默认 `0` | Trace store 数据库故障时是否允许退回进程内存 |
| `AGENT_TRACE_MEMORY_TTL_SECONDS` | `3600` | dev 内存审计缓存 TTL |
| `AGENT_TRACE_MEMORY_MAX_ITEMS` | `200` | dev 内存审计缓存上限 |
| `VIX_CIRCUIT_BREAKER_THRESHOLD` | `30` | 风险熔断阈值 |

### 下游服务地址

| 变量 | 默认值 |
| --- | --- |
| `FORECAST_URL` | `http://localhost:8010/api/v1/forecast` |
| `MEMORY_URL` | `http://localhost:8012/api/v1/memory/search` |
| `MARKET_SNAPSHOT_URL` | `http://localhost:8014/api/v1/market/snapshot/latest` |
| `RECENT_NEWS_URL` | `http://localhost:8016/api/v1/news/recent` |

### 数据与回退

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/postgres` | Postgres 连接串 |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis 连接串 |
| `MARKET_ALLOW_SYNTHETIC_FALLBACK` | `1` | 行情失败时生成样本快照 |
| `NEWS_ALLOW_SAMPLE_FALLBACK` | `1` | 新闻失败时回退缓存或样本流 |
| `MARKET_START_BACKGROUND_TASK` | `0` | 本地调试默认关闭后台刷新 |
| `NEWS_START_BACKGROUND_TASK` | `0` | 本地调试默认关闭后台刷新 |
| `NEWS_FETCH_TIMEOUT_SECONDS` | `4.0` | 新闻抓取超时 |
| `NEWS_STALE_AFTER_SECONDS` | `300` | 新闻陈旧阈值 |
| `NEWS_STALE_CACHE_GRACE_SECONDS` | `1800` | 陈旧缓存可接受窗口 |
| `INFERENCE_ALLOW_SYNTHETIC_FALLBACK` | `1` | 量化预测无法拉取原始输入时退回启发式代理 |

### OpenAI 叙事层

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 空 | 可选；为空时使用规则回退 |
| `AGENT_DEFAULT_MODEL` | `gpt-5.4-mini` | 默认叙事模型 |
| `AGENT_COMPLEX_MODEL` | `gpt-5.4` | 复杂场景叙事模型 |

LLM 只负责叙事增强和输出组织，不承担证据检索、风险熔断或事实存储职责。

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

## 模型与回退行为

### 量化预测

`inference_service.py` 优先加载 checkpoint 进行真实预测；当模型不可用、输入准备失败或市场数据无法正常取得时，会退回启发式代理结果，并在响应中把 `forecast_basis` 标成 `heuristic_proxy`。

这意味着：

- `T+1 / T+7` 优先来自模型
- `T+30` 当前用于中期参考，不应当被解读为独立训练的长期预测系统
- 服务在不满足条件时倾向于“保守可用”，而不是“强行自信”

### 市场与新闻

- `market_snapshot_service.py` 可以在源故障时输出 `synthetic_fallback`
- `news_ingest_service.py` 可以在源故障时回退缓存，再不行才退回样本新闻
- `news` 和 `memory` 的服务契约都带有 `status`、`degraded_reason`、`source_freshness_seconds`

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

### 2. 反馈入口

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

### 3. 调试 / 审计入口

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

### 4. 内部 QA 入口

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

CI 当前包含两层保障：

- Python 3.12 下的正式测试集
- 基于 `docker compose` 的最小主链路冒烟

## 生产部署建议

- 把 `APP_ENV` 设为 `production`
- 显式配置 `AGENT_PUBLIC_API_KEYS` 和 `AGENT_INTERNAL_API_KEYS`
- 仅允许受信来源访问 `traces` 与 `trigger`
- 在反向代理或 API Gateway 层补充 TLS、IP 约束和速率限制
- 使用真实 Postgres / Redis，不依赖 dev 内存 trace fallback
- 将 OpenAI key 视为可选增强，而不是系统可用性的前提

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

### 没有 `OPENAI_API_KEY` 会不会直接不可用

不会。系统会退回确定性规则生成，接口仍然可用，只是文案表达会更保守。

## 补充说明

- 所有 FastAPI 服务在直接启动时，默认都可通过 `/docs` 查看 OpenAPI 页面。
- 如果你只需要验证正式主链路，请优先使用 `analyze`、`feedback` 和 `traces`，不要把 `trigger` 当成对外契约。
