# AI Agent Token 优化与成本控制方案

**文档版本**: 1.0  
**负责人**: 微软 AI Agent 架构师 (Persona)  
**目标**: 在保持 90% 以上响应质量的前提下，降低 80% 的 Token 消耗。

---

## 1. 现状分析与高消耗环节识别

基于 `test_llm_service_fixed.py` 定义的架构（Router-Generator-Consolidator）及 `data_loader.py` 的业务逻辑，预估以下环节为 Token 消耗大户：

| 环节 | 原始模式 (High Cost) | 消耗预估 (每请求) | 优化目标 |
| :--- | :--- | :--- | :--- |
| **新闻分析** | 将每日 50+ 条原始新闻全文输入 LLM 进行情感打分 | 15k - 30k tokens | **< 1k tokens** |
| **决策简报** | 输入所有历史价格、技术指标和完整新闻上下文生成报告 | 5k - 8k tokens | **< 800 tokens** |
| **多轮对话** | 在 Chat 模式下携带完整历史对话记录（无限上下文） | 线性增长 (10k+) | **固定窗口 < 2k** |
| **工具调用** | 为了容错，在 System Prompt 中包含所有工具的详细 Schema | 2k - 3k tokens | **动态加载 < 500** |

---

## 2. 核心优化策略

### 2.1 轻量级提示词工程 (Prompt Engineering)

*   **结构化输出 (JSON Mode)**: 强制模型输出 JSON，减少“废话”和过渡性文本。
*   **Reference-based Generation**: 仅提供关键差异数据，而非全量背景。
*   **模板示例**:
    ```python
    # Bad Prompt
    "请阅读以下所有新闻，并分析它们对金价的影响，写一篇详细的报告..." (附带 100 条新闻)

    # Optimized Prompt
    "基于以下 3 个关键市场事件（已筛选），简述对 XAU/USD 的 24h 趋势影响 (涨/跌/震荡)。格式: JSON。"
    ```

### 2.2 智能上下文压缩 (Context Compression)

*   **RAG (检索增强生成)**: 不再将所有新闻放入 Context。先将新闻存入向量库 (ChromaDB/FAISS)，根据用户问题检索 Top-3 相关片段。
*   **摘要链 (Map-Reduce)**:
    1.  **Map**: 使用极速模型 (GPT-4o-mini) 对每条长新闻生成 50 字摘要。
    2.  **Reduce**: 将摘要聚合，输入主模型生成最终观点。
*   **动态剪枝**: 移除相关性得分低于 0.7 的上下文片段。

### 2.3 模型分层策略 (Model Routing)

利用 `ModelRole` 架构实现“大材大用，小材小用”：

*   **Router (路由层)**: 使用 `GPT-4o-mini` 或 `Claude-3-Haiku`。负责意图识别，消耗极低。
*   **Generator (生成层)**:
    *   简单任务 (如数据查询): `GPT-4o-mini`。
    *   复杂推理 (如趋势预测): `GPT-4o` / `Claude-3.5-Sonnet`。
*   **Consolidator (整合层)**: 使用 `GPT-4o-mini` 快速格式化输出。

---

## 3. 函数调用与工具优化

*   **动态工具加载**: 不要一次性把 50 个工具的 definition 发给 LLM。
    *   第一步：Router 判断意图是 "查询价格" 还是 "分析新闻"。
    *   第二步：仅加载该意图相关的 3-5 个工具定义。
*   **参数简化**: 工具参数尽量设计为枚举值 (Enum) 或布尔值，减少 LLM 推理参数格式的 Token 消耗。

---

## 4. Token 预算与配额管理

制定严格的 Token 预算策略：

*   **Global Limit**: 每日总消耗上限 (例如 $5.00)。
*   **Per-Request Limit**:
    *   **L1 (简单查询)**: Max 500 tokens (In+Out)
    *   **L2 (深度分析)**: Max 4000 tokens (In+Out)
*   **熔断机制**: 当单次对话 Context 超过 8k 时，强制触发“总结与遗忘”机制，将前文压缩为 200 字摘要。

---

## 5. 监控与告警体系

实施 `TokenUsageTracker` 类，实时监控：

1.  **Total Tokens**: 累计消耗。
2.  **Cost Estimated**: 估算金额。
3.  **Efficiency Ratio**: (输出字符数 / 输入 Token 数)，越低说明 Prompt 越冗余。

**告警阈值**:
*   单次请求 > 10k tokens -> **Warning** (钉钉/Slack 通知)
*   每分钟请求成本 > $0.5 -> **Critical** (暂停非核心服务)

---

## 6. A/B 测试验证计划

*   **对照组 (Control)**: 原始逻辑，直接 Prompt 全量上下文，使用 GPT-4。
*   **实验组 (Experiment)**: 实施 RAG + 摘要压缩 + 混合模型 (Mini+4o)。
*   **评估指标**:
    1.  **Token 消耗量**: 目标降低 80%。
    2.  **响应准确性**: 人工评分 (1-5分) 或 LLM-as-a-Judge 打分。
    3.  **响应延迟**: P95 延迟降低 50%。

---

## 7. 实施指南 (Next Steps)

1.  创建 `ratelmind/services/token_optimizer.py`：实现压缩与追踪逻辑。
2.  重构 `llm_service.py`：集成 Router 逻辑与动态工具加载。
3.  更新 `pro_dashboard.py`：接入优化后的 API。

建议立即开始 **Step 1: 创建 Token 优化器工具库**。
