# P3+P4: RAG 知识库(事件研究)+ 新闻精准反应 Implementation Plan

> REQUIRED SUB-SKILL: superpowers:executing-plans

**Goal:** 学术级事件研究库(永久层)+ 决策相关新闻精选档案(180 天衰减层)+ 混合检索;
事件分类→历史类比先验→委员会 news 分析师升级→重大事件缓存失效。

**Architecture:**
- `knowledge/events_catalog.jsonl`: 人工精选 2004-2025 重大宏观事件(类别/日期/标题/摘要),入库为不可变语料。
- `event_study.py`: 用扩展数据集计算每事件后 5/30/90 天金价前向收益;按类别聚合(均值/中位/n/方向占比);TTL 缓存。
- `news_archive.py`: RSS 项 → 决策相关性规则打分(类别关键词+信息增量启发式)→ 去重 → JSONL 存档,180 天 TTL,检索按相关性×时间衰减排序。纯本地,无新依赖;嵌入/pgvector 为升级路径。
- `event_classifier.py`(P4): 文本 → {category, severity};规则可审计。
- `knowledge_retriever.py`: 事件类比(经 classifier 定类)+ 新闻档案关键词双路,输出带 citation。
- 网关: news_analyst 升级(情绪+历史类比先验);高严重度新闻→research_context 缓存失效钩子;`GET /api/v1/agent/knowledge/search` 端点。

**Global Constraints:** 事件目录中的历史统计必须由 event_study 从真实价格数据计算,禁止手写收益数字;新闻档案只留决策相关;一切降级显式;CI 离线全绿。

### Task 1: 事件目录 + event_study.py(TDD)
- events_catalog.jsonl: ≥40 条,字段 {event_id, date, category ∈ {monetary_policy, inflation, geopolitics, usd, flows}, title, summary}
- `load_catalog() -> List[Dict]`(校验字段/类别/日期格式)
- `compute_event_study(catalog, prices) -> Dict`: per-event {fwd_5d, fwd_30d, fwd_90d}(事件日后首个交易日为锚,不足样本→None);per-category 聚合 {n, mean_30d, median_30d, positive_share_30d, ...}
- `class EventStudyLibrary`: TTL 缓存 + `analogs_for(category) -> Dict`(聚合+最近 5 条事件引用)
- 测试: 合成价格上手算前向收益;类别聚合;目录 schema 校验;真实目录加载全通过

### Task 2: event_classifier.py(TDD)
- `classify_news(text) -> {"category", "severity", "matched"}`;规则:类别关键词表(中英),严重度=命中强词(war/紧急/collapse/暴跌/加息 75bp 等)→high,普通命中→medium,无→low/None
- 测试: 各类别样文;严重度分档;无关文本→None

### Task 3: news_archive.py(TDD)
- `relevance_score(item) -> float`(0-1: 类别命中+强词+来源权重);`DECISION_THRESHOLD`
- `NewsArchive(path)`: `ingest(items) -> {kept, dropped}`(打分过滤+标题规范化去重+append);`search(query, now) -> List`(关键词命中×exp 时间衰减,180 天过期不返回);`prune(now)`(TTL 物理清理)
- 测试: 过滤丢噪音;去重;TTL 淘汰;检索排序含衰减

### Task 4: knowledge_retriever.py + 网关端点(TDD)
- `search_knowledge(query, *, event_library, news_archive) -> {"event_analogs", "news_hits", "citations"}`
- `GET /api/v1/agent/knowledge/search?q=`(public)
- 测试: 端点契约;citation 字段齐全;组件缺失→显式降级

### Task 5: news_analyst 升级 + 缓存失效(P4 核心,TDD)
- `news_analyst(news_sentiment, vix_value, *, analog_prior=None)`: 类比先验(mean_30d 方向×positive_share 强度,有界 ±0.3)与情绪分融合;evidence 引用"历史上 n 次类似事件后 30 天平均 x%"
- gateway analyze 路径: 对 optional_news_text/最新新闻走 classifier → EventStudyLibrary.analogs_for → 传入委员会;trace 记录 knowledge_analogs
- `research_context.invalidate()` 方法 + 高严重度新闻钩子(news ingest 侧标记,gateway 读到 high severity → force_refresh 下次生效)
- 测试: 委员会 news 观点受先验影响(有界);trace 含类比;invalidate 后重算

### Task 6: 回归 + README + 提交
