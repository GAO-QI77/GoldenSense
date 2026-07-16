# GoldenSense 双 Agent 投研系统 — 设计文档

日期: 2026-07-16
状态: 已获用户批准
分支: codex/dual-agent-system (基于 codex/goldensense-public-demo)

## 1. 目标

在已有的分层量化研究层(HAR-RV / HMM+四因子 / 公允价值+情景锥 / 分析师委员会 / 校准闭环)
之上,完成五个产品目标:

1. **Agent①「大盘观点书」**: 短/中/长三时间尺度的统一系统观点,含每周不可变信号台账。
2. **Agent②「个性化研究」**: 按投资者画像(风险承受力/持仓/期限/经验)生成定制化研究分析,
   规则引擎产数字 + DeepSeek 产叙事。
3. **RAG 知识库**: 学术级事件研究库(永久层)+ 决策相关新闻精选档案(180 天衰减层),混合检索,引用可溯源。
4. **新闻精准反应**: 事件分类 → 严重度 → 历史类比先验 → 委员会 news 分析师升级 + 重大事件缓存失效触发。
5. **首页整合**: 市场快照 + 观点书摘要 + 本周信号 + 个性化入口。

## 2. 已确认的关键决策

| 决策 | 结论 |
|---|---|
| 资产范围 | 黄金为主 + 宏观背景因子(维持 GoldenSense 定位,不扩多资产) |
| 产品定性 | **个性化研究分析,非投顾**。输出为"参考区间/差距/风险提示"框架,禁止指令式建议措辞;全输出带免责 |
| 用户体系 | 轻量画像,无登录。画像存 localStorage,随请求携带,服务端不持久化个人数据 |
| 信号台账 | Append-only JSONL(git 可追踪,离线可跑),持久化注入 seam 供生产换 Postgres;每周一发布,幂等,不可变,不回填 |
| 台账计分 | 影子组合持三档中点,现金为非黄金腿,计成本(bps),仅对已成熟周计分;对照固定基准(静态权重 + 金价买入持有) |
| RAG 语料 | 历史事件+事件研究法标注(永久) + 新闻精选(决策相关性过滤 + 去重 + 180 天 TTL + 时间衰减) |
| LLM | DeepSeek(key 已配置),复用现有 narrator 基建;一切 LLM 输出过 narrative_critic;降级路径显式 |

## 3. 纪律红线(继承项目 DNA,不可违反)

1. LLM 永不产生数字,只组织语言;critic 数字校验不通过 → 回退确定性模板并打 `narrative_critic_reverted`。
2. 任何新信号/数字结论必须来自已验证的量化模块(allocation/vol_models/fair_value/HMM/committee)。
3. 台账 append-only + content hash;历史记录永不重写;回测与前向追踪严格分离标注。
4. 所有外部依赖(DeepSeek/RSS/pgvector/网络)必须有离线降级,CI 无网全绿。
5. 委员会分歧只驱动叙事模型路由与置信度,不接管 stance 门控(历史教训,见 memory)。
6. 服务测试注入 `_MemoryPersistence` 隔离本机 Redis/Postgres。

## 4. 子项目设计

### P1 — Agent①「大盘观点书」+ 每周信号台账

**新模块 `market_view.py`**: 从 `research_context.shared_context` 组装 `MarketViewBook`:

- `short_term`(1-21天): vol_bands(h1/h5/h21)+ 已实现波动状态。表述为区间与风险,
  不做方向预测(方向预测已被走前验证证伪)。
- `mid_term`(1-6月): HMM regime_posterior + macro_factors 复合 + 委员会 fused_stance/分歧度。
- `long_term`(6月+): fair_value 偏离(z 分/regime_break)+ scenario_cone + flagship 当前目标敞口。
- 每节字段: `core_view`(核心判断)、`confidence`、`evidence`(证据链)、
  `invalidation`(失效条件——什么数据变化会使该观点作废)。
- 顶部: `data_asof/data_age_days/degraded` 诚实透传。

**新模块 `signal_ledger.py`**: 每周不可变发布:

- `SignalPublication`: publication_id(`YYYY-Www`)、published_at、data_asof、
  allocations(三档 range+midpoint)、market_view 摘要、evidence_snapshot、degraded、content_hash(sha256)。
- 存储: `data_cache/signal_ledger.jsonl` append-only;`LedgerStore` 抽象类 + JSONL 实现(生产可换 DB)。
- 发布幂等: 同一 ISO 周重复发布返回已有记录,不覆盖。
- 计分 `score_track_record()`: 每档影子组合按 midpoint 持金、余现金,发布日调仓,
  换手成本默认 5bps;仅对下一发布日价格已存在的周计分;输出累计收益/波动/最大回撤/Sharpe
  vs 两个固定基准。
- 调度: 复用 `scripts/refresh_data.py` 的调度模式,周一发布;也提供 CLI 手动发布。

**端点**: `GET /api/v1/agent/market-view`;`GET /api/v1/signals/current|history|track-record`,
`GET /api/v1/signals/{publication_id}`。

### P2 — Agent②「个性化研究」

**`investor_profile.py`**: Pydantic 画像模型:
`risk_tolerance ∈ {conservative, balanced, aggressive}`、`horizon ∈ {short, mid, long}`、
`current_gold_pct ∈ [0,100]`、`experience ∈ {novice, experienced, professional}`。
严格校验,extra=forbid。

**`personal_research.py`**: 确定性规则引擎,输入(画像, MarketViewBook, allocation, RAG 类比),输出结构化 JSON:

- `reference_range`: 画像风险档对应的 allocation 区间(复用 allocation.py,不新算)。
- `position_gap`: 当前仓位 vs 区间(below/within/above + 幅度 pct)。
- `risk_flags`: 规则触发的结构化提示,如
  - 仓位高于区间上沿 且 P(stress) 超阈值 → `position_above_range_in_stress`
  - 短期画像 且 21 天波动带宽超阈值 → `short_horizon_high_vol`
  - 期限与观点书证据错配 → `horizon_mismatch`
- `horizon_evidence`: 按期限挑选观点书对应小节作为主证据。
- 每个结论附 `evidence_ref`(指向观点书/allocation 字段路径)。

**DeepSeek 叙事层**: 复用 narrator 基建;prompt 含结构化事实 + 经验等级(决定解释深度)+
RAG 历史类比;输出过 narrative_critic;新增**去指令化 judge**(eval/judges): 禁止
"你应该买入/卖出/加仓"类措辞,必须保持"参考区间/差距/风险提示"框架;免责声明必须出现。
critic 或 judge 不通过 → 确定性模板降级。

**端点**: `POST /api/v1/agent/personal-research`(画像在 body,不持久化)。

**前端**: `/advisor` 页——画像表单(localStorage)+ 结果视图(区间/差距/风险旗标/叙事/免责)。

### P3 — RAG 知识库

**事件研究库(永久层)**: `knowledge/events_catalog.jsonl` 人工精选 2004-2026 重大宏观事件
(FOMC 转向、CPI 意外、地缘冲突、央行购金、ETF 大进出),字段: 日期/类别/标题/摘要。
`event_study.py` 用扩展数据集计算每事件后 5/30/90 天前向收益,并按类别聚合
(均值/中位/样本量/命中方向占比)。检索命中 → "历史上 n 次类似事件后 30 天平均 +x%(n=..)"。

**新闻精选档案(衰减层)**: `news_archive.py`:

- 决策相关性两级过滤: 规则分类器(类别关键词 + 信息增量启发式)打分;边界分数段批量走
  DeepSeek 判别(有降级: 无 key 时只留规则高分段)。
- 去重(标题规范化 + 近似匹配),180 天 TTL 淘汰,检索时按时间衰减加权。
- 存储: pgvector 复用 memory_ingestion 的 schema 模式 + 无 DB 时的内存/JSONL 降级。

**混合检索 `knowledge_retriever.py`**: 向量(MiniLM)+ 关键词双路召回,合并去重,
每条结果带 `{source, date, type: event|news, citation}`。无嵌入模型/DB 时降级为关键词检索。

### P4 — 新闻精准反应

- `event_classifier.py`: 类别(monetary_policy/inflation/geopolitics/usd/flows)×
  严重度(low/medium/high),规则为主(可审计),关键词+来源权重。
- 影响映射: 类别 → 事件研究库类比集 → 前向收益先验 → **升级 `news_analyst`**:
  stance 由"情绪分 + 历史类比先验"共同决定,evidence 引用类比统计。
- 重大事件触发: 高严重度 → `research_context` 缓存失效 + 前端横幅"重大事件,观点书更新中"。

### P5 — 首页整合

- 首页: 市场快照(现有 service,补前端呈现 + 延迟标注)+ 观点书三尺度摘要卡 +
  本周信号卡 + `/advisor` 入口。
- 新页: `/advisor`、`/signals`(台账 + 追踪记录图)。
- ⌘K searchIndex 覆盖新页面/面板。黑金主题延续。

## 5. 测试与验收

- 每模块单测: 台账不可变/幂等/计分成熟逻辑;规则引擎全分支;critic/judge 硬门;
  过滤器/分类器;检索降级路径。
- 端点契约测试注入内存持久化。
- eval golden set 新增个性化用例,judges 加去指令化 + 免责在场,进 CI 硬门。
- 浏览器实测所有新页面;`npm run build` 通过。
- 全量 pytest 保持全绿(204 → 预计 ~260)。

## 6. 交付顺序

1. P1 观点书 + 台账(端点 + 测试)
2. P2 个性化 Agent(DeepSeek 真实调用 + 评测硬门)
3. P3 RAG + P4 新闻反应
4. P5 前端整合 + 全量回归 + 文档

每期结束提交一次,保持可回退。
