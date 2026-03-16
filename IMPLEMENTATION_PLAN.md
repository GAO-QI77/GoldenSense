# GoldenSense 改进实施方案 (Implementation Plan)

**基于**: AI Agent 评估问卷与改进路线图 (v1.0)  
**目标**: 打造高可用、真实可信、安全合规的金融 AI Agent 系统  
**版本**: 1.0

---

## 1. 核心架构重构 (Architecture Refactoring) - P0

### 1.1 任务描述
将前端展示（FastAPI/Streamlit）与 AI 推理逻辑解耦，移除演示用的随机噪声，接入真实的 `StackingModel` 进行实时预测。

### 1.2 涉及模块
*   `live_platform/server.py`: 后端服务入口
*   `stacking_model.py`: AI 模型定义与推理逻辑
*   `train_stacking.py`: (辅助) 模型训练与保存

### 1.3 实施步骤
1.  **模型持久化适配**: 修改 `stacking_model.py`，增加 `save_model()` 和 `load_model()` 方法，支持保存训练好的权重（XGBoost, Torch models, Ridge weights）。
2.  **推理服务集成**:
    *   在 `server.py` 启动时 (`startup_event`) 加载预训练模型。
    *   如果无预训练模型，初始化一个未训练的模型实例（仅用于演示架构连通性，并在日志中警告）。
3.  **真实预测管道**:
    *   在 `fetch_market_data` 中，除了获取价格，还需构建特征工程（调用 `feature_engineer.py`）。
    *   将实时特征输入模型，获取 `1d`, `7d`, `30d` 的预测结果。
    *   更新 `market_cache`，存储真实的预测值而非随机数。

### 1.4 验证计划
*   **单元测试**: 测试 `load_model` 能否成功加载权重文件。
*   **集成测试**: 启动 `server.py`，访问 `/api/stream`，验证返回的数据字段是否与模型输出一致（不再是随机波动）。

---

## 2. 数据可靠性与容错 (Data Reliability) - P1

### 2.1 任务描述
增强 `yfinance` 数据抓取的稳定性，增加重试机制与异常处理，防止单点故障导致服务崩溃。

### 2.2 涉及模块
*   `live_platform/server.py`

### 2.3 实施步骤
1.  **指数退避重试**: 引入 `tenacity` 库或手写重试逻辑，当 `yf.download` 失败时，等待 1s, 2s, 4s 后重试。
2.  **异常捕获兜底**:
    *   如果重试 3 次仍失败，保持 `market_cache` 中上一次的有效数据，并标记 `status: stale`。
    *   前端接收到 `stale` 状态时，显示黄色警告图标 "Data Outdated"。
3.  **数据清洗**: 在存入 Cache 前，检查价格是否为 `NaN` 或 `0`，过滤异常跳变。

### 2.4 验证计划
*   **故障模拟测试**: 断开网络连接，观察控制台日志是否进行重试，前端是否显示 "Data Outdated"。

---

## 3. 安全与合规 (Security & Compliance) - P0

### 3.1 任务描述
实施基础的 API 安全策略，移除敏感信息，并确保法律合规。

### 3.2 涉及模块
*   `live_platform/server.py`
*   `live_platform/static/index.html`

### 3.3 实施步骤
1.  **API Key 鉴权**:
    *   在 `server.py` 中定义 `API_KEY` (从环境变量读取)。
    *   添加 FastAPI `Depends` 依赖，检查请求头 `X-API-Key`。
    *   (注：对于公共演示页，可保留 `/` 和 `/api/stream` 的匿名访问，但限制高频 API `/api/history` 的访问)。
2.  **CORS 策略收紧**: 将 `allow_origins=["*"]` 改为配置化列表，生产环境仅允许部署域名。
3.  **免责声明**: 在 `index.html` 底部 Footer 区域增加显著的免责声明文本（双语）。

### 3.4 验证计划
*   **安全扫描**: 使用 `curl` 不带 Token 访问受保护接口，确认返回 403 Forbidden。

---

## 4. 性能优化 (Performance) - P1

### 4.1 任务描述
提升静态资源加载速度与 API 响应效率。

### 4.2 涉及模块
*   `live_platform/server.py`
*   `live_platform/static/index.html`

### 4.3 实施步骤
1.  **Gzip 压缩**: 在 FastAPI 中添加 `GZipMiddleware(minimum_size=1000)`。
2.  **CDN 加速**: 确认 `index.html` 引用的 Tailwind 和 Chart.js 使用了可靠的 CDN (如 cdnjs.cloudflare.com)。
3.  **SSE 频率优化**: 将 `message_stream` 的推送频率从 1s 调整为 3s（平衡实时性与服务器负载）。

### 4.4 验证计划
*   **性能测试**: 使用 Chrome DevTools Network 面板查看 `content-encoding: gzip` 响应头。

---

## 5. 迭代任务清单 (Task List)

| 优先级 | 任务ID | 任务名称 | 预估工时 | 风险点 | 回滚策略 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P0** | TASK-01 | **接入真实模型推理** | 4h | 模型加载内存溢出 | 回退至随机数生成逻辑 |
| **P0** | TASK-02 | **添加免责声明与 Token 清理** | 1h | 无 | N/A |
| **P1** | TASK-03 | **数据抓取重试机制** | 2h | 重试导致主线程阻塞 | 禁用重试，恢复单次请求 |
| **P1** | TASK-04 | **开启 Gzip 与性能调优** | 1h | CPU 占用率微升 | 移除中间件配置 |
| **P2** | TASK-05 | **前端断线重连优化** | 2h | 客户端重连风暴 | 恢复旧版 JS 逻辑 |

---

## 6. 执行指令 (Next Steps)

建议优先执行 **TASK-01** 和 **TASK-02**，确保提交给 Hackathon 的版本是“真实”且“合规”的。

如果您同意，我可以立即开始执行 **TASK-01**，修改 `server.py` 以集成 `stacking_model.py`。
