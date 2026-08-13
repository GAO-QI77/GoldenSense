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

若需要先发布可体验版本，再补齐第三方服务，可以暂时省略 `DEEPSEEK_API_KEY`、`DATABASE_URL` 和 `REDIS_URL`。启动器会使用规则叙事、内存 trace 和显式降级结果。Neon、Upstash 与 DeepSeek 接入后，无需改代码，只需补充环境变量并重新部署。

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
