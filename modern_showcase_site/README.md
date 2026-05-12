# GoldenSense Consumer Web

这是 GoldenSense 面向中文散户的消费者前台，不再是通用展示模板。

它现在承担三件事：

- `/` 展示黄金预测、四类指标、新闻、引用和数据质量。
- `/agent` 接收完整风险问卷、用户问题、风险偏好和分析周期。
- 渲染 Agent 返回的风险适配 briefing、证据卡、引用、自动抓取的新闻和三周期判断。

## 开发

```bash
npm install
npm run dev
```

默认会通过 Vite 开发代理请求：

```bash
/api/v1/agent/dashboard/current
/api/v1/agent/analyze
/api/v1/agent/feedback
```

代理默认会转发到：

```bash
http://127.0.0.1:8020
```

如需直连或覆盖：

```bash
VITE_AGENT_API_URL=http://localhost:8020/api/v1/agent/analyze
VITE_AGENT_DASHBOARD_URL=http://localhost:8020/api/v1/agent/dashboard/current
VITE_AGENT_FEEDBACK_URL=http://localhost:8020/api/v1/agent/feedback
VITE_AGENT_API_KEY=dev-public-key
VITE_AGENT_PROXY_TARGET=http://127.0.0.1:8020
```

## 构建

```bash
npm run build
npm run test:e2e
```

`test:e2e` 使用 Playwright 覆盖首页指标、Agent 完整问卷和移动端横向溢出检查。

## 页面结构

- 研究主页：价格条、三周期预测、四类指标、数据质量、自动新闻和指标引用。
- Agent 页：完整风险问卷、风险适配建议、证据卡、新闻、引用和反馈。
- 结论区反馈：用户可以直接标记“有帮助 / 没帮助”，把 `analysis_id` 回传给后端；请求会附带低权限 public API key。

## 设计原则

- 面向中文散户研究辅助场景，用词清楚但保持专业密度。
- 不做“神谕式”喊单，所有结论都必须有失效条件、引用或降级标记。
- 风格是安静、密集、克制的黄金研究终端，不做营销落地页。
