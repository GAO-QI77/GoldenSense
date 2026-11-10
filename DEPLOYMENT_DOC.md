# GoldenSense Agent 部署指南

本仓库当前只保留正式 GoldenSense Agent 主链路，不再包含旧的直播平台或旧 Streamlit 看板部署方式。

## 1. 基线环境

- Python 3.12
- Node.js 20
- Docker / Docker Compose（可选）

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

## 2. 本地启动后端主链路

推荐直接使用：

```bash
zsh scripts/dev_stack.sh start
```

查看状态：

```bash
zsh scripts/dev_stack.sh status
```

停止服务：

```bash
zsh scripts/dev_stack.sh stop
```

默认会拉起：

- `inference_service.py`
- `memory_service.py`
- `market_snapshot_service.py`
- `news_ingest_service.py`
- `agent_gateway.py`

本地脚本默认不加载 embedding 模型，因此 `memory_service` 可以 HTTP-ready 但会在响应体中显示
`status=unavailable / retriever_status=not_started`。这是开发环境显式降级，不是已加载的记忆服务。完整验收前应执行
`memory_ingestion.py` 并设置 `MEMORY_START_BACKGROUND_LOAD=1`。

## 3. 鉴权配置

正式 Agent 网关要求 API key：

```bash
export AGENT_PUBLIC_API_KEYS=dev-public-key
export AGENT_INTERNAL_API_KEYS=dev-internal-key
```

- `GET /api/v1/agent/dashboard/current`、`POST /api/v1/agent/analyze` 与 `POST /api/v1/agent/feedback`：接受 public 或 internal key
- `GET /api/v1/agent/traces/{analysis_id}` 与 `POST /api/v1/agent/trigger`：只接受 internal key

除 `development` 之外，服务会拒绝缺失 API key 或继续使用 `dev-public-key` / `dev-internal-key` 的启动配置。

## 4. 前端启动

消费者前台：

```bash
cd modern_showcase_site
npm install
npm run dev
```

常用前端环境变量：

```bash
VITE_AGENT_API_URL=http://localhost:8020/api/v1/agent/analyze
VITE_AGENT_DASHBOARD_URL=http://localhost:8020/api/v1/agent/dashboard/current
VITE_AGENT_FEEDBACK_URL=http://localhost:8020/api/v1/agent/feedback
VITE_AGENT_API_KEY=dev-public-key
```

内部 QA / 运营面板：

```bash
python3 -m streamlit run frontend/dashboard.py
```

如需走旧的内部触发入口，请配置：

```bash
export AGENT_GATEWAY_INTERNAL_API_KEY=dev-internal-key
```

## 4.1 Vercel 前端 + Railway 单容器后端

推荐公开散户助手时只把 `modern_showcase_site/` 部署到 Vercel，并把 Python 服务栈部署到 Railway 单容器。不要从仓库根目录执行 Vercel 发布；根目录包含模型 checkpoint、研究数据和后端源码。

### Railway

仓库根目录包含 [`railway.json`](railway.json)。Railway 会构建 Dockerfile，并运行：

```bash
python3 scripts/public_stack.py
```

启动器只公开 `${PORT}` 上的 Agent Gateway。`inference`、`memory`、`market` 和 `news` 服务只监听容器内的 `127.0.0.1`。
公网镜像使用 [`requirements.public.txt`](requirements.public.txt)，并从 PyTorch 官方 CPU 索引安装 Torch，避免 Railway CPU 容器下载 CUDA 运行时。

Railway Demo 环境变量：

```bash
APP_ENV=demo
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-key
AGENT_DEFAULT_MODEL=deepseek-v4-flash
AGENT_COMPLEX_MODEL=deepseek-v4-pro
LLM_TIMEOUT_SECONDS=12.0
AGENT_PUBLIC_API_KEYS=replace-with-random-public-key
AGENT_INTERNAL_API_KEYS=replace-with-random-internal-key
AGENT_ALLOW_ORIGINS=https://your-vercel-domain.vercel.app
DATABASE_URL=postgresql://... # Neon pooled connection string
REDIS_URL=rediss://...        # Upstash Redis connection string
```

可直接从 [`railway.demo.env.example`](railway.demo.env.example) 开始填写。示例文件不包含真实密钥。

`APP_ENV=demo` 会显式使用 `yfinance + RSS`，并保留带标记的 fallback。该模式适合公开黑客松体验，不应描述为交易级行情基础设施。

公开环境需要可访问的 PostgreSQL 持久化账本和审计连接。账本默认使用 `DATABASE_URL`，也可以通过 `SIGNAL_LEDGER_DATABASE_URL` 指定。LLM 和 Redis 仍可按项目配置选择，缺少 LLM 时用规则叙事。

### 成本基线

| 服务 | 首发配置 | 费用基线 |
| --- | --- | --- |
| Vercel | 静态前端 | 免费层 |
| Railway | Hobby 单容器 | `$5/月`，包含 `$5` 资源用量 |
| Neon | Free Postgres / pgvector | `$0`，每项目 `100 CU-hours/月`、`0.5 GB` |
| Upstash | Free Redis | `$0`，`256 MB`、`500K commands/月` |
| DeepSeek | 按量计费 | 小额余额，关闭无限自动充值 |

首发继续使用 `yfinance + RSS`。若后续需要更稳定的黄金现货 API，优先评估 Metals.Dev Silver：`$9.99/月`、`10,000 requests/月`，配合 5 分钟缓存。

DeepSeek 默认模型为 `deepseek-v4-flash`，高风险场景为 `deepseek-v4-pro`。不要新接入 `deepseek-chat` 或 `deepseek-reasoner`：官方已标注这两个兼容别名将在 `2026-07-24 15:59 UTC` 弃用。

### Neon / pgvector

创建 Neon 数据库后执行一次历史记忆初始化：

```bash
python3 memory_ingestion.py \
  --database-url "$DATABASE_URL"
```

脚本会在 Neon 支持的情况下自动创建 `vector` 扩展；否则回退到数组存储。

### Vercel

当前公开前端地址：

```text
https://goldensense-public-site.vercel.app
```

当前 Railway Demo 后端地址：

```text
https://agent-gateway-production-fa79.up.railway.app
```

前端环境变量：

```bash
VITE_AGENT_API_URL=https://agent-gateway-production-fa79.up.railway.app/api/v1/agent/analyze
VITE_AGENT_DASHBOARD_URL=https://agent-gateway-production-fa79.up.railway.app/api/v1/agent/dashboard/current
VITE_AGENT_FEEDBACK_URL=https://agent-gateway-production-fa79.up.railway.app/api/v1/agent/feedback
VITE_AGENT_API_KEY=your-public-key
```

Vercel 的 SPA 路由由 `modern_showcase_site/vercel.json` 处理，浏览器按上述 URL 直接访问 Railway。`VITE_` 变量会进入前端构建产物，内部 API key 和数据库凭证应留在后端环境。

后端生产环境至少需要：

```bash
APP_ENV=production
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-deepseek-key
AGENT_PUBLIC_API_KEYS=your-public-key
AGENT_INTERNAL_API_KEYS=your-internal-key
AGENT_ALLOW_ORIGINS=https://your-vercel-domain.vercel.app
MARKET_DATA_PROVIDER=your-market-provider
NEWS_DATA_PROVIDER=your-news-provider
MARKET_ALLOW_SYNTHETIC_FALLBACK=0
NEWS_ALLOW_SAMPLE_FALLBACK=0
INFERENCE_ALLOW_SYNTHETIC_FALLBACK=0
AGENT_ALLOW_TRACE_MEMORY_FALLBACK=0
```

生产模式应接入经过授权的行情与新闻供应商。它和黑客松 `APP_ENV=demo` 的免费源回退策略是两个不同的运行档位。

ResearchCase 生产发布还需确认：

- 镜像包含 `tesseract-ocr` 和 `tesseract-ocr-chi-sim`；不可用时图片分析必须显式 `abstain`。
- `RESEARCH_CASE_LEDGER_PATH` 位于持久卷或换成等价持久化实现。JSONL为追加式修订记录，不覆盖历史。
- 账本仅保存文档哈希、证据定位、事实卡、门控和研究结果；原始上传文件与投资者画像不落盘。
- URL输入每次重定向重新执行公网HTTPS校验，拒绝私网、回环、链路本地和保留地址。
- 前端为案件请求发送随机 `X-Research-Session`；服务端按会话哈希隔离案件，不能跨会话读取或个性化。
- 通过内部 cron 调用 `POST /api/v1/agent/research-cases/score-due`，完成到期 checkpoint 前向评分。
- 前端和下游消费者必须读取 `probability_kind`：`research_weight` 只能展示为研究权重，不能标注为预测概率或胜率。
- 个性化消费者必须尊重 `rules.suitability.position_analysis_allowed`；为 `false` 时不得自行恢复仓位差距或目标比例。

发布前检查：

```bash
curl https://agent-gateway-production-fa79.up.railway.app/health/live
curl https://agent-gateway-production-fa79.up.railway.app/health/ready
```

## 5. Docker Compose

```bash
docker compose up --build
```

当前编排会启动：

- `redis`
- `postgres`
- `inference`
- `memory`
- `market_ingest`
- `news_ingest`
- `agent_gateway`
- `frontend`
- `webapp`

## 6. 验证

核心测试：

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

主链路冒烟：

```bash
python3 scripts/smoke_agent.py
```

比赛硬门验收（GOAI 官方六项权重，包含 100 并发、四页联通、三期限、攻击阻断和公开输出审计）：

```bash
python3 scripts/competition_acceptance.py \
  --gateway-url http://127.0.0.1:8020 \
  --frontend-url http://127.0.0.1:4173
```

首页聚合默认使用 15 秒共享快照合并并发下游请求。读、分析、研究、个性化和写入使用独立限流桶；429 响应包含
`Retry-After`、中文原因和重试时间，前端不得将其呈现为研究结论。

## 7. 研究数据与数据库升级

在 Railway 持久卷上保存研究数据，确保应用和定时任务使用同一组文件路径。Docker 构建会排除本地 `data_cache/`，部署时需单独准备这些数据。

| 变量 | 文件用途 |
| --- | --- |
| `GOLD_RESEARCH_MARKET_PATH` | 量化研究、事件研究和影子评分共用的扩展行情 |
| `MACRO_VINTAGES_PATH` | 带可见时间的 ALFRED 宏观历史版本 |
| `GOLD_DRIVER_FLOWS_PATH` | 带可见时间的 ETF、COT 和央行购金数据 |
| `GOLD_RISK_REVIEW_PATH` | 短期候选风险模型的复验摘要 |

`python3 scripts/refresh_data.py` 刷新扩展行情，获取失败时保留上一份有效数据。
Railway 单容器默认在启动时刷新并按天继续运行，可用
`DATA_REFRESH_INTERVAL_SECONDS` 调整间隔。过期快照保留用于带标记的研究降级，当周发布等待数据恢复。启用 `GOLD_DRIVER_COT_REFRESH_ENABLED=1`
后，刷新任务同时采集 COT。宏观版本单独运行 `python3 scripts/refresh_macro_vintages.py` 更新，
可在执行环境配置 `FRED_API_KEY` 或用 `--alfred-exports` 导入核验过的离线数据。

短期波动复验、分析校准和影子组合可使用同一套经核验的黄金会话数据。
将价格文件 `Date,Gold` 和会话文件 `Date,close_at,available_at` 放在持久卷，
按日期逐行对应。两个时间列都必须含时区。配置 `GOLD_VERIFIED_CLOSES_PATH`
和 `GOLD_VERIFIED_SESSIONS_PATH` 后，影子记分与分析校准只读截止前已可见的报价。
原有日级代理仍可用于研究，记分卡会显示该口径。用正式数据跑短期复验时，将
两份文件传给 `scripts/evaluate_short_term_risk.py` 并显式指定评估起点和截止时间。

在迁移任务的运行环境配置 `DATABASE_URL`，先在测试数据库验证，再对目标数据库运行只读预检，
确认历史事件表、新闻快照表和待清理审计记录的状态。确认目标与备份后可执行可重复运行的结构迁移，
再以只读预检核对结果。Railway 使用 `/health/ready` 检查下游和账本可用性，结构迁移和旧审计清理单独执行。

```bash
python3 scripts/migrate_release.py
python3 scripts/migrate_release.py --apply-schema
python3 scripts/migrate_release.py
```

旧审计数据清理会永久精简旧载荷、清空工具调用和证据详情、删除反馈自由文本，并仅保留脚本允许的期限、
摘要立场、置信度等级及反馈评分。确认留存要求和备份后，单独执行清理并检查返回的剩余条数。

```bash
python3 scripts/migrate_release.py --redact-legacy
```

影子发布依赖当期研究数据。遇到过期或缺失数据时，系统保留已经登记的记录。
使用 `python3 scripts/publish_signal.py --no-digest` 可仅登记当前周，不发送摘要。
