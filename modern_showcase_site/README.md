# GoldenSense Consumer Web

这是 GoldenSense 面向中文散户的消费者前台，不再是通用展示模板。

它现在承担三件事：

- 展示“现在怎么看黄金”的结论卡。
- 接收用户问题、风险偏好和分析周期，不再要求手动输入新闻。
- 渲染 Agent 返回的证据卡、引用、自动抓取的新闻和三周期判断。

## 开发

```bash
npm install
npm run dev
```

默认会通过 Vite 开发代理请求：

```bash
/api/v1/agent/analyze
```

代理默认会转发到：

```bash
http://127.0.0.1:8020
```

反馈默认会请求：

```bash
/api/v1/agent/feedback
```

如需直连或覆盖：

```bash
VITE_AGENT_API_URL=http://localhost:8020/api/v1/agent/analyze
VITE_AGENT_FEEDBACK_URL=http://localhost:8020/api/v1/agent/feedback
VITE_AGENT_API_KEY=dev-public-key
VITE_AGENT_PROXY_TARGET=http://127.0.0.1:8020
```

## 构建

```bash
npm run build
```

已验证当前版本可以成功生成 `dist/`。

## 页面结构

- 顶部 Hero：一句话结论、风险条和当前画像。
- 中部工作区：提问表单 + 自动新闻预览 + 对话与建议。
- 下部证据区：三周期参考、证据卡、可追溯引用和后续追问建议。
- 结论区反馈：用户可以直接标记“有帮助 / 没帮助”，把 `analysis_id` 回传给后端；请求会附带低权限 public API key。

## 设计原则

- 面向无技术背景散户，用词短、结论清楚。
- 不做“神谕式”喊单，所有结论都必须有失效条件。
- 风格以金色、羊皮纸和深墨色为主，不走默认 SaaS 模板路线。
