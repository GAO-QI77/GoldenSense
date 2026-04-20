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

- `POST /api/v1/agent/analyze` 与 `POST /api/v1/agent/feedback`：接受 public 或 internal key
- `GET /api/v1/agent/traces/{analysis_id}` 与 `POST /api/v1/agent/trigger`：只接受 internal key

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
