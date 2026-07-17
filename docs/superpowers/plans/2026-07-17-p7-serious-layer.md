# P7: 严重层修复 Implementation Plan(工业运行标准)

> REQUIRED SUB-SKILL: superpowers:executing-plans

**Goal:** 修复 PM 评审严重层:①留存闭环(邮件周报订阅)②渐进披露(简明/专业模式)③RAG 扩容与自动归档 + 记忆服务接线文档 ④跨资产对照(单资产天花板第一步)⑤运营健康检查告警。

**Global Constraints:** 邮箱等个人数据只存本地 data_cache(gitignored),带退订令牌;一切外部通道(SMTP/webhook)可缺省降级为日志并显式标注;测试与 .env 隔离(沿用 conftest);pytest/eval/e2e/build 全绿。

### S1 邮件周报订阅(留存)
- `subscriptions.py`: `SubscriptionStore`(JSONL data_cache/subscriptions.jsonl,email 规范化去重,secrets token,active 标记,unsubscribe(token));`render_digest(publication, track_record) -> {subject, text}`(三档区间+观点书摘要+防篡改哈希+免责+退订占位);`send_digest(...)` 传输层:SMTP_HOST 配置则 smtplib,否则 log-only 返回 {"delivered": false, "reason": "smtp_not_configured"}(诚实降级)
- 端点: `POST /api/v1/subscriptions` {email}(public key+限流,返回 masked email);`GET /api/v1/subscriptions/unsubscribe?token=`(无需 key,邮件链接可点,幂等)
- 接线: scripts/publish_signal.py publish_once 成功且 created=True 时发送周报(可 `--no-digest`);网关 autopublish 同样
- 前端: SignalsPage 订阅卡(邮箱输入+成功/退订态)
- 测试: store 去重/规范化/退订幂等;digest 渲染含区间与免责;无 SMTP 时 delivered=false;端点契约含 422 邮箱校验

### S4 跨资产对照
- research_context 新块 `cross_asset`: 对 [Silver, S&P500, USD_Index, Crude_Oil, 10Y_Bond] 计算 63d 滚动相关(末值)+ 1y 累计收益对照(黄金 vs 各资产);缺列显式降级
- 测试: 合成数据手算相关/收益;缺列降级
- 前端: QuantPage 或 SignalsPage 增「跨资产背景」面板(相关色条+1y 对照)

### S5 健康检查告警
- `scripts/health_check.py`: 检查 gateway /health、research_context data_age(>4d 告警)、/metrics 5xx 率、本周 signals 是否已发布;失败项汇总 → exit 1;ALERT_WEBHOOK_URL 配置则 POST JSON(urllib,超时 5s,失败不抛);`--json` 输出
- 测试: 注入 fake fetcher 各失败态;webhook 降级

### S3 RAG 扩容 + 自动归档 + 记忆接线
- events_catalog 52 → 120+(真实事件,市场反应日;类别覆盖检查测试不变)
- 新闻自动归档: refresh_once 尾部追加 news_archive 摄取(从 news_ingest_service 的 fetch 管道取最近项,失败不影响主流程,summary 记 kept/dropped);语料自此自动累积
- 记忆服务: README 运维节补 docker compose up postgres + memory_ingestion 一键指引(不强制本机起 docker)
- 测试: refresh summary 含 archive 字段(注入 fake fetch)

### S2 渐进披露(简明/专业)
- 顶栏 ViewMode 切换(简明/专业,localStorage gs_view_mode,默认简明)
- 简明模式: 首页隐藏指标内部行/来源健康/引用面板(保留市场条/核心结论/预测卡/走势/观点书/新闻);SignalsPage 隐藏证据行/哈希/历史表(保留三档区间+核心观点+记分卡);QuantPage 顶部提示「专业页面」
- e2e: 既有断言涉及被隐藏元素的用例先置专业模式(localStorage 预置);新增简明模式用例
- 测试: e2e 全绿 + 浏览器实测两种模式
