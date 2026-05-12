import React, { useEffect, useMemo, useRef, useState } from 'react';
import { BrowserRouter, Link, NavLink, Route, Routes } from 'react-router-dom';
import {
  AlertTriangle,
  ArrowRight,
  BadgeCheck,
  BarChart3,
  BookOpenCheck,
  BrainCircuit,
  CalendarDays,
  Calculator,
  CheckCircle2,
  ChevronRight,
  Crosshair,
  Clock3,
  DatabaseZap,
  ExternalLink,
  FileSearch,
  Gauge,
  Landmark,
  LineChart,
  Loader2,
  LockKeyhole,
  Newspaper,
  Radar,
  ShieldCheck,
  SlidersHorizontal,
  Target,
  TrendingDown,
  TrendingUp,
  WalletCards,
} from 'lucide-react';

const API_URL = import.meta.env.VITE_AGENT_API_URL || '/api/v1/agent/analyze';
const DASHBOARD_URL =
  import.meta.env.VITE_AGENT_DASHBOARD_URL || API_URL.replace('/analyze', '/dashboard/current');
const FEEDBACK_URL = import.meta.env.VITE_AGENT_FEEDBACK_URL || API_URL.replace('/analyze', '/feedback');
const API_KEY = import.meta.env.VITE_AGENT_API_KEY || 'dev-public-key';

const horizonLabels = {
  '24h': '短线 T+1',
  '7d': '中线 T+7',
  '30d': '长线 T+30',
};

const horizonShortLabels = {
  '24h': 'T+1',
  '7d': 'T+7',
  '30d': 'T+30',
};

const riskLabels = {
  conservative: '保守型',
  balanced: '平衡型',
  aggressive: '进取型',
};

const stanceClass = {
  偏多: 'tone-bull',
  偏空: 'tone-bear',
  中性: 'tone-neutral',
  高风险观望: 'tone-risk',
};

const groupIcons = {
  fundamental: Landmark,
  technical: LineChart,
  macro_policy: Gauge,
  flow_sentiment: Radar,
};

const indicatorDirectionLabel = {
  bullish: '偏多',
  bearish: '偏空',
  neutral: '中性',
  risk: '风险',
};

const positionLabels = {
  none: '无持仓',
  long: '已有多头',
  short: '已有空头',
  hedged: '已对冲',
};

const baseCapital = 100000;

const investorDefaults = {
  risk_capacity: 'medium',
  trading_horizon: 'short',
  experience_level: 'intermediate',
  capital_allocation_pct: 10,
  max_drawdown_pct: 8,
  current_position: 'none',
  liquidity_need: 'medium',
  leverage_attitude: 'none',
  investment_goal: 'event_trade',
};

const selectMeta = {
  risk_capacity: [
    ['low', '低'],
    ['medium', '中'],
    ['high', '高'],
  ],
  trading_horizon: [
    ['short', '短线'],
    ['medium', '中线'],
    ['long', '长线'],
  ],
  experience_level: [
    ['beginner', '新手'],
    ['intermediate', '有经验'],
    ['advanced', '成熟交易者'],
  ],
  current_position: [
    ['none', '无持仓'],
    ['long', '已有多头'],
    ['short', '已有空头'],
    ['hedged', '已对冲'],
  ],
  liquidity_need: [
    ['low', '低'],
    ['medium', '中'],
    ['high', '高'],
  ],
  leverage_attitude: [
    ['none', '不用杠杆'],
    ['low', '低杠杆'],
    ['medium', '中等杠杆'],
    ['high', '高杠杆'],
  ],
  investment_goal: [
    ['capital_preservation', '本金保护'],
    ['income', '稳健增值'],
    ['event_trade', '事件交易'],
    ['trend_following', '趋势跟随'],
    ['speculation', '投机博弈'],
  ],
};

const starterPrompts = [
  '如果今晚 CPI 高于预期，黄金短线应该如何控制风险？',
  '美元指数继续走强时，黄金 T+7 的失效条件是什么？',
  '我已经有黄金多头，接下来一周该关注哪些指标？',
  '地缘冲突升温但 ETF 没有流入，黄金是不是只适合观望？',
];

function apiHeaders(extra = {}) {
  return {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
    ...extra,
  };
}

async function readApiJson(response, fallbackMessage) {
  const text = await response.text();
  const trimmed = text.trim();
  let json = null;

  if (trimmed) {
    try {
      json = JSON.parse(trimmed);
    } catch (error) {
      throw new Error(`${fallbackMessage}：服务返回非 JSON 内容。`);
    }
  }

  if (!response.ok) {
    const detail = json?.detail;
    const message = typeof detail === 'string' ? detail : detail?.message || json?.message;
    throw new Error(message || `${fallbackMessage}：HTTP ${response.status}`);
  }

  if (!json) {
    throw new Error(`${fallbackMessage}：服务返回空响应。`);
  }

  return json;
}

function forecastBasisLabel(forecast) {
  if (!forecast) return '等待预测基线';
  if (forecast.basis === 'heuristic_proxy') return '代理量化引擎：真实行情驱动';
  if (forecast.basis === 'degraded_fallback' || forecast.model_status === 'unavailable') return '量化引擎不可用';
  if (forecast.basis === 'ensemble_model') return '训练量化模型';
  return forecast.basis || '预测基线';
}

function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/agent" element={<AgentPage />} />
          <Route path="*" element={<DashboardPage />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

function AppShell({ children }) {
  return (
    <div className="terminal-shell">
      <header className="topbar">
        <Link to="/" className="brand-lockup" aria-label="GoldenSense home">
          <span className="brand-mark">
            <BarChart3 size={18} />
          </span>
          <span>
            <strong>GoldenSense</strong>
            <small>Gold Research Terminal</small>
          </span>
        </Link>

        <nav className="topnav" aria-label="Main navigation">
          <NavLink to="/" end>
            <LineChart size={16} />
            研究主页
          </NavLink>
          <NavLink to="/agent">
            <BrainCircuit size={16} />
            风险画像 Agent
          </NavLink>
        </nav>

        <div className="compliance-pill">
          <ShieldCheck size={15} />
          研究辅助 · 非下单系统
        </div>
      </header>
      {children}
    </div>
  );
}

function DashboardPage() {
  const [dashboard, setDashboard] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    async function loadDashboard() {
      try {
        setLoading(true);
        setError('');
        const response = await fetch(DASHBOARD_URL, {
          method: 'GET',
          headers: apiHeaders(),
          signal: controller.signal,
        });
        const json = await readApiJson(response, '首页研究数据读取失败');
        setDashboard(json);
      } catch (loadError) {
        if (loadError.name !== 'AbortError') {
          setError(loadError.message || '首页研究数据读取失败');
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }
    loadDashboard();
    return () => controller.abort();
  }, []);

  const market = dashboard?.market_status;
  const forecasts = dashboard?.horizon_forecasts || [];
  const groups = dashboard?.indicator_groups || [];
  const dataQuality = dashboard?.data_quality;
  const degradationFlags = dashboard?.degradation_flags || [];
  const news = dashboard?.recent_news || [];
  const citations = dashboard?.citations || [];
  const sourceHealth = dashboard?.source_health || [];
  const goldHistory = dashboard?.gold_history;

  const primaryForecast = forecasts[0];
  const dashboardStatus = loading ? 'loading' : error ? 'error' : dataQuality?.status || 'ok';
  const dashboardInsights = useMemo(
    () => buildDashboardInsights({ market, forecasts, groups, news, dataQuality, degradationFlags }),
    [market, forecasts, groups, news, dataQuality, degradationFlags],
  );

  return (
    <main className="page-surface dashboard-page">
      <section className="terminal-header">
        <div>
          <p className="eyebrow">XAUUSD Research Brief</p>
          <h1>黄金价格预测与指标总览</h1>
          <p>
            主页只展示稳定市场基线和指标证据；个人风险画像、周期选择和适配建议放在独立 Agent 页处理。
          </p>
        </div>
        <StatusBadge status={dashboardStatus} loading={loading} />
      </section>

      {error ? <ErrorPanel title="首页研究数据不可用" message={error} /> : null}

      <section className="market-strip" aria-label="Market summary">
        <MetricTile
          label="XAUUSD"
          value={loading ? '读取中' : market ? formatPrice(market.latest_price) : 'N/A'}
          detail={market ? `1D ${formatPercent(market.price_change_pct_1d)}` : '等待市场快照'}
          tone={market?.price_change_pct_1d >= 0 ? 'bull' : market?.price_change_pct_1d < 0 ? 'bear' : 'neutral'}
          icon={market?.price_change_pct_1d >= 0 ? TrendingUp : TrendingDown}
        />
        <MetricTile
          label="主预测"
          value={primaryForecast ? primaryForecast.stance : loading ? '读取中' : 'N/A'}
          detail={primaryForecast ? `${primaryForecast.action} · ${primaryForecast.confidence_band}置信度` : 'T+1 baseline'}
          tone={primaryForecast ? toneName(primaryForecast.stance) : 'neutral'}
          icon={Radar}
        />
        <MetricTile
          label="数据新鲜度"
          value={market ? `${market.freshness_seconds}s` : loading ? '读取中' : 'N/A'}
          detail={market?.is_stale ? '快照陈旧' : dataQuality?.status === 'degraded' ? '含降级数据' : '当前可用'}
          tone={market?.is_stale || dataQuality?.status === 'degraded' ? 'risk' : 'bull'}
          icon={DatabaseZap}
        />
        <MetricTile
          label="质量提示"
          value={degradationFlags.length ? `${degradationFlags.length} 项` : loading ? '读取中' : '正常'}
          detail={degradationFlags[0] || dataQuality?.indicator_status || '无降级标记'}
          tone={degradationFlags.length ? 'risk' : 'neutral'}
          icon={BadgeCheck}
        />
      </section>

      <section className="research-brief-grid">
        <CoreThesisPanel thesis={dashboardInsights.thesis} />
        <DriverMatrix groups={groups} drivers={dashboardInsights.drivers} />
      </section>

      <section className="forecast-grid" aria-label="Forecast horizons">
        {forecasts.length ? (
          forecasts.map((forecast) => <ForecastCard key={forecast.horizon} forecast={forecast} />)
        ) : (
          <PlaceholderPanel icon={Clock3} text={loading ? '正在读取 T+1 / T+7 / T+30 预测基线。' : '暂无预测基线。'} />
        )}
      </section>

      <GoldTrendPanel history={goldHistory} loading={loading} />

      <section className="research-brief-grid lower">
        <CatalystCalendar events={dashboardInsights.events} />
        <ScenarioPanel scenarios={dashboardInsights.scenarios} />
      </section>

      <section className="workspace-layout">
        <div className="primary-column">
          <SectionHeader
            kicker="Indicator Pillars"
            title="四类核心指标"
            description="基本面、技术面、宏观政策和资金情绪分开展示，每个指标保留来源、状态与新鲜度。"
          />
          <div className="indicator-grid">
            {groups.length ? (
              groups.map((group) => <IndicatorGroupCard key={group.id} group={group} />)
            ) : (
              <PlaceholderPanel icon={Gauge} text={loading ? '正在读取指标柱。' : '暂无指标数据。'} />
            )}
          </div>
        </div>

        <aside className="side-rail">
          <QualityPanel quality={dataQuality} flags={degradationFlags} />
          <SourceHealthPanel sources={sourceHealth} loading={loading} />
          <NewsPanel news={news} loading={loading} />
          <CitationPanel citations={citations} />
          <Link className="agent-entry" to="/agent">
            <span>
              <strong>进入风险画像 Agent</strong>
              <small>填写完整问卷后生成风险适配 briefing</small>
            </span>
            <ArrowRight size={17} />
          </Link>
        </aside>
      </section>
      <TerminalFooter
        left="GoldenSense Research Dashboard"
        right="价格、指标、来源健康与风险提示统一在首页收口"
      />
    </main>
  );
}

function AgentPage() {
  const [question, setQuestion] = useState(starterPrompts[0]);
  const [riskProfile, setRiskProfile] = useState('balanced');
  const [horizon, setHorizon] = useState('24h');
  const [investorProfile, setInvestorProfile] = useState(investorDefaults);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const resultRef = useRef(null);

  const summary = analysis?.summary_card;
  const riskBanner = analysis?.risk_banner;
  const forecasts = analysis?.horizon_forecasts || [];
  const evidenceCards = analysis?.evidence_cards || [];
  const citations = analysis?.citations || [];
  const recentNews = analysis?.recent_news || [];
  const selectedForecast = forecasts.find((item) => item.horizon === horizon) || forecasts[0];

  const profileScore = useMemo(() => {
    let score = 0;
    if (Number(investorProfile.capital_allocation_pct) >= 50) score += 3;
    else if (Number(investorProfile.capital_allocation_pct) >= 25) score += 2;
    else if (Number(investorProfile.capital_allocation_pct) >= 10) score += 1;
    if (Number(investorProfile.max_drawdown_pct) <= 5) score += 2;
    else if (Number(investorProfile.max_drawdown_pct) <= 10) score += 1;
    score += { none: 0, low: 1, medium: 2, high: 3 }[investorProfile.leverage_attitude] || 0;
    if (investorProfile.experience_level === 'beginner') score += 1;
    if (investorProfile.liquidity_need === 'high') score += 2;
    if (['long', 'short'].includes(investorProfile.current_position)) score += 1;
    if (investorProfile.investment_goal === 'speculation') score += 1;
    return score;
  }, [investorProfile]);

  const profileLevel = profileScore >= 5 ? '高' : profileScore >= 2 ? '中' : '低';
  const riskBudget = useMemo(() => buildRiskBudget(investorProfile, profileScore), [investorProfile, profileScore]);
  const suitabilityGate = useMemo(
    () => buildSuitabilityGate(investorProfile, profileScore, riskBudget),
    [investorProfile, profileScore, riskBudget],
  );
  const executionScenarios = useMemo(
    () => buildExecutionScenarios({ summary, selectedForecast, riskBudget, investorProfile }),
    [summary, selectedForecast, riskBudget, investorProfile],
  );

  function updateInvestorProfile(key, value) {
    setInvestorProfile((current) => ({
      ...current,
      [key]: ['capital_allocation_pct', 'max_drawdown_pct'].includes(key) ? Number(value) : value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      setError('请输入具体问题后再开始分析。');
      return;
    }

    setLoading(true);
    setError('');
    setFeedbackStatus('');

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: apiHeaders(),
        body: JSON.stringify({
          question: trimmed,
          risk_profile: riskProfile,
          horizon,
          locale: 'zh-CN',
          investor_profile: investorProfile,
        }),
      });
      const json = await readApiJson(response, 'Agent 分析失败');
      setAnalysis(json);
      window.setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    } catch (submitError) {
      setError(submitError.message || 'Agent 分析失败');
    } finally {
      setLoading(false);
    }
  }

  async function submitFeedback(rating) {
    if (!analysis?.analysis_id) return;
    try {
      const response = await fetch(FEEDBACK_URL, {
        method: 'POST',
        headers: apiHeaders(),
        body: JSON.stringify({ analysis_id: analysis.analysis_id, rating, comment: null }),
      });
      await readApiJson(response, '反馈提交失败');
      setFeedbackStatus(rating === 'helpful' ? '已记录：这条 briefing 有帮助。' : '已记录：这类回答会进入后续评估。');
    } catch (feedbackError) {
      setFeedbackStatus(feedbackError.message || '反馈提交失败');
    }
  }

  return (
    <main className="page-surface agent-page">
      <section className="terminal-header">
        <div>
          <p className="eyebrow">Retail Suitability Workflow</p>
          <h1>风险画像 Agent</h1>
          <p>
            先收集完整问卷，再把稳定预测解释成风险适配建议。输出包含证据、失效条件和禁止执行条件，不提供下单指令。
          </p>
        </div>
        <div className={`profile-score level-${profileLevel === '高' ? 'high' : profileLevel === '中' ? 'medium' : 'low'}`}>
          <span>问卷风险</span>
          <strong>{profileLevel}</strong>
          <small>score {profileScore}</small>
        </div>
      </section>

      <section className="agent-layout">
        <form className="agent-form" onSubmit={handleSubmit}>
          <PanelTitle icon={SlidersHorizontal} title="分析输入" subtitle="问题、周期和基础风险偏好" />

          <label className="field">
            <span>你的问题</span>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={6}
              placeholder="例：如果 CPI 高于预期，黄金短线和一周视角分别要怎么看？"
            />
          </label>

          <div className="prompt-bank">
            {starterPrompts.map((prompt) => (
              <button key={prompt} type="button" onClick={() => setQuestion(prompt)}>
                {prompt}
              </button>
            ))}
          </div>

          <div className="control-pair">
            <SegmentedControl
              label="风险承受能力"
              value={riskProfile}
              options={[
                ['conservative', riskLabels.conservative],
                ['balanced', riskLabels.balanced],
                ['aggressive', riskLabels.aggressive],
              ]}
              onChange={setRiskProfile}
            />
            <SegmentedControl
              label="分析周期"
              value={horizon}
              options={Object.entries(horizonLabels)}
              onChange={setHorizon}
            />
          </div>

          <PanelTitle icon={WalletCards} title="完整风险问卷" subtitle="用于限制建议强度，不改变市场预测基线" />

          <div className="questionnaire-grid">
            <SelectField
              label="风险容量"
              value={investorProfile.risk_capacity}
              options={selectMeta.risk_capacity}
              onChange={(value) => updateInvestorProfile('risk_capacity', value)}
            />
            <SelectField
              label="交易周期"
              value={investorProfile.trading_horizon}
              options={selectMeta.trading_horizon}
              onChange={(value) => updateInvestorProfile('trading_horizon', value)}
            />
            <SelectField
              label="经验水平"
              value={investorProfile.experience_level}
              options={selectMeta.experience_level}
              onChange={(value) => updateInvestorProfile('experience_level', value)}
            />
            <SelectField
              label="已有持仓"
              value={investorProfile.current_position}
              options={selectMeta.current_position}
              onChange={(value) => updateInvestorProfile('current_position', value)}
            />
            <NumberField
              label="资金占比"
              suffix="%"
              value={investorProfile.capital_allocation_pct}
              min="0"
              max="100"
              onChange={(value) => updateInvestorProfile('capital_allocation_pct', value)}
            />
            <NumberField
              label="最大回撤"
              suffix="%"
              value={investorProfile.max_drawdown_pct}
              min="0"
              max="100"
              onChange={(value) => updateInvestorProfile('max_drawdown_pct', value)}
            />
            <SelectField
              label="流动性需求"
              value={investorProfile.liquidity_need}
              options={selectMeta.liquidity_need}
              onChange={(value) => updateInvestorProfile('liquidity_need', value)}
            />
            <SelectField
              label="杠杆态度"
              value={investorProfile.leverage_attitude}
              options={selectMeta.leverage_attitude}
              onChange={(value) => updateInvestorProfile('leverage_attitude', value)}
            />
            <SelectField
              label="投资目标"
              value={investorProfile.investment_goal}
              options={selectMeta.investment_goal}
              onChange={(value) => updateInvestorProfile('investment_goal', value)}
            />
          </div>

          <RiskBudgetPanel budget={riskBudget} profile={investorProfile} />
          <SuitabilityGatePanel gate={suitabilityGate} />

          {error ? <ErrorPanel title="Agent 分析失败" message={error} /> : null}

          <button className="primary-action" type="submit" disabled={loading}>
            {loading ? <Loader2 size={17} className="spinning" /> : <BrainCircuit size={17} />}
            {loading ? '生成 briefing 中' : '生成风险适配 briefing'}
          </button>
        </form>

        <aside className="agent-context">
          <PanelTitle icon={LockKeyhole} title="Agent 边界" subtitle="工业级研究终端的输出纪律" />
          <ul className="boundary-list">
            <li>预测基线来自行情和量化服务，不被用户问题改写。</li>
            <li>问卷只影响风险适配、建议强度和禁止执行条件。</li>
            <li>数据陈旧、证据冲突、提示注入或过度确定性语言会强制降级。</li>
            <li>所有结论必须能回到证据卡、引用或降级标记。</li>
          </ul>
          {selectedForecast ? (
            <div className="sticky-forecast">
              <span>{horizonShortLabels[selectedForecast.horizon]}</span>
              <strong>{selectedForecast.stance}</strong>
              <small>{forecastBasisLabel(selectedForecast)}</small>
              <p>{selectedForecast.reasons?.[0]}</p>
            </div>
          ) : (
            <div className="sticky-forecast muted">
              <span>等待分析</span>
              <strong>暂无 briefing</strong>
              <p>提交后这里会保留当前周期的稳定预测基线。</p>
            </div>
          )}
          <PositionModePanel budget={riskBudget} profile={investorProfile} />
        </aside>
      </section>

      <section ref={resultRef} className="analysis-output">
        <SectionHeader
          kicker="Agent Briefing"
          title="风险适配分析"
          description="输出方向、情景、失效条件和禁止执行条件，避免把研究解读误读成下单指令。"
        />

        {!analysis && !loading ? (
          <PlaceholderPanel icon={BookOpenCheck} text="填写问卷并提交后，这里会展示完整 briefing、证据卡和引用。" />
        ) : null}

        {loading ? (
          <PlaceholderPanel icon={Loader2} text="Agent 正在读取预测基线、新闻、历史类比和问卷门控。" spinning />
        ) : null}

        {summary ? (
          <div className="briefing-grid">
            <section className="decision-panel">
              <div className="decision-head">
                <span className={`stance-badge ${stanceClass[summary.stance] || 'tone-neutral'}`}>{summary.stance}</span>
                <span>{summary.confidence_band}置信度</span>
              </div>
              <h2>{summary.action}</h2>
              <p>{summary.reasons?.[0]}</p>
              <div className={`risk-banner level-${riskBanner?.level || 'medium'}`}>
                <strong>{riskBanner?.title}</strong>
                <span>{riskBanner?.message}</span>
              </div>
              <div className="reason-columns">
                <SummaryList title="关键理由" items={summary.reasons || []} />
                <SummaryList title="禁止执行 / 失效条件" items={summary.invalidators || []} />
              </div>
              <p className="disclaimer">{summary.disclaimer}</p>
              <div className="feedback-row">
                <span>这条 briefing 是否有帮助？</span>
                <button type="button" onClick={() => submitFeedback('helpful')}>
                  有帮助
                </button>
                <button type="button" onClick={() => submitFeedback('not_helpful')}>
                  没帮助
                </button>
              </div>
              {feedbackStatus ? <p className="feedback-status">{feedbackStatus}</p> : null}
            </section>

            <section className="evidence-panel">
              <PanelTitle icon={FileSearch} title="证据卡片" subtitle={`${evidenceCards.length} 张证据 / ${citations.length} 条引用`} />
              <div className="evidence-list">
                {evidenceCards.map((card) => (
                  <article key={card.id} className={`evidence-card ${card.direction}`}>
                    <span>{card.signal_type}</span>
                    <h3>{card.title}</h3>
                    <p>{card.takeaway}</p>
                    <small>{card.citation_ids.map((id) => `#${id}`).join(' ')}</small>
                  </article>
                ))}
              </div>
            </section>
          </div>
        ) : null}

        {summary ? <ExecutionScenarioPanel scenarios={executionScenarios} /> : null}

        {analysis ? (
          <section className="post-analysis-grid">
            <div className="panel-block">
              <PanelTitle icon={Newspaper} title="本轮新闻" subtitle={`${recentNews.length} 条`} />
              <div className="compact-list">
                {recentNews.length ? (
                  recentNews.slice(0, 4).map((item) => (
                    <article key={item.event_id}>
                      <span>{item.source} · {formatDateTime(item.published_at)}</span>
                      <strong>{item.title}</strong>
                      <p>{item.summary}</p>
                    </article>
                  ))
                ) : (
                  <p>本轮没有可展示的新闻条目。</p>
                )}
              </div>
            </div>
            <div className="panel-block">
              <PanelTitle icon={ShieldCheck} title="可追溯引用" subtitle={`${citations.length} 条`} />
              <div className="compact-list">
                {citations.map((citation) => (
                  <article key={citation.id}>
                    <span>{citation.id} · {citation.source_type}</span>
                    <strong>{citation.label}</strong>
                    <p>{citation.excerpt}</p>
                    {citation.url ? (
                      <a href={citation.url} target="_blank" rel="noreferrer">
                        打开来源
                        <ExternalLink size={13} />
                      </a>
                    ) : null}
                  </article>
                ))}
              </div>
            </div>
          </section>
        ) : null}
      </section>
      <TerminalFooter
        left="GoldenSense Agent Workbench"
        right="问卷、门控、证据与引用统一在 Agent 页收口"
      />
    </main>
  );
}

function CoreThesisPanel({ thesis }) {
  return (
    <section className={`thesis-panel tone-${thesis.tone}`}>
      <PanelTitle icon={Target} title="今日核心结论" subtitle={thesis.kicker} />
      <div className="thesis-body">
        <strong>{thesis.headline}</strong>
        <p>{thesis.summary}</p>
      </div>
      <div className="thesis-brief-list">
        <MetricLine label="最强支撑" value={thesis.support} />
        <MetricLine label="最大压制" value={thesis.pressure} />
        <MetricLine label="关键失效" value={thesis.invalidator} />
      </div>
    </section>
  );
}

function DriverMatrix({ drivers }) {
  return (
    <section className="driver-panel">
      <PanelTitle icon={Crosshair} title="驱动贡献矩阵" subtitle="四类指标对黄金方向的净贡献" />
      <div className="driver-list">
        {drivers.map((driver) => (
          <article key={driver.id} className={`driver-row ${driver.tone}`}>
            <div>
              <strong>{driver.title}</strong>
              <span>{driver.summary}</span>
            </div>
            <div className="driver-meter" aria-label={`${driver.title} contribution ${driver.score}`}>
              <i style={{ width: `${Math.max(8, Math.abs(driver.score) * 100)}%` }} />
            </div>
            <strong>{driver.scoreLabel}</strong>
          </article>
        ))}
      </div>
    </section>
  );
}

function CatalystCalendar({ events }) {
  return (
    <section className="catalyst-panel">
      <PanelTitle icon={CalendarDays} title="风险催化日历" subtitle="未来几天最容易改变黄金定价的事件" />
      <div className="catalyst-list">
        {events.map((event) => (
          <article key={event.name} className={`catalyst-item impact-${event.impact}`}>
            <span>{event.window}</span>
            <strong>{event.name}</strong>
            <p>{event.watch}</p>
            <small>{event.impactLabel}</small>
          </article>
        ))}
      </div>
    </section>
  );
}

function ScenarioPanel({ scenarios }) {
  return (
    <section className="scenario-panel">
      <PanelTitle icon={BookOpenCheck} title="三情景推演" subtitle="把单点预测拆成可观察路径" />
      <div className="scenario-grid">
        {scenarios.map((scenario) => (
          <article key={scenario.name} className={`scenario-card ${scenario.tone}`}>
            <span>{scenario.name}</span>
            <strong>{scenario.path}</strong>
            <p>{scenario.trigger}</p>
            <small>{scenario.invalidator}</small>
          </article>
        ))}
      </div>
    </section>
  );
}

function GoldTrendPanel({ history, loading }) {
  const points = history?.points || [];
  const keyNodes = history?.key_nodes || [];
  const chart = useMemo(() => buildTrendChart(points, keyNodes), [points, keyNodes]);
  const sourceLabel = history?.source || '等待数据';
  const sourceTone = sourceLabel.includes('fallback') || sourceLabel.includes('synthetic') ? '降级行情源' : '真实行情源';

  return (
    <section className="gold-trend-panel">
      <div className="trend-panel-head">
        <PanelTitle icon={LineChart} title="黄金价格走势" subtitle={`${sourceTone} · ${sourceLabel}`} />
        <div className="trend-stat-strip">
          <MetricLine label="样本点" value={points.length ? `${points.length}` : loading ? '读取中' : 'N/A'} />
          <MetricLine label="关键波动节点" value={`${keyNodes.length}`} />
        </div>
      </div>

      {points.length >= 2 ? (
        <div className="trend-chart-layout">
          <div className="trend-chart-shell" aria-label="Gold price trend chart">
            <svg viewBox="0 0 920 280" role="img" aria-label="黄金价格走势折线图">
              <defs>
                <linearGradient id="goldTrendFill" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="rgba(201,154,56,0.32)" />
                  <stop offset="100%" stopColor="rgba(201,154,56,0)" />
                </linearGradient>
              </defs>
              <path className="trend-area" d={chart.areaD} />
              <path className="trend-line" d={chart.pathD} />
              {chart.keyMarkers.map((marker) => (
                <g key={`${marker.date}-${marker.price}`} className={`trend-marker ${marker.direction}`}>
                  <line x1={marker.x} x2={marker.x} y1="24" y2="252" />
                  <circle cx={marker.x} cy={marker.y} r="7">
                    <title>{marker.reason}</title>
                  </circle>
                  <text x={marker.x} y={Math.max(18, marker.y - 14)} textAnchor="middle">
                    {formatPercent(marker.change_pct)}
                  </text>
                </g>
              ))}
            </svg>
            <div className="trend-axis">
              <span>{points[0]?.date}</span>
              <strong>{formatPrice(chart.latestPrice)}</strong>
              <span>{points[points.length - 1]?.date}</span>
            </div>
          </div>
          <div className="key-node-panel">
            <strong>关键波动节点</strong>
            <div className="key-node-list">
              {keyNodes.length ? (
                keyNodes.slice(0, 5).map((node) => (
                  <article key={`${node.date}-${node.price}`} className={`key-node-card ${node.direction}`}>
                    <span>{node.date} · {formatPercent(node.change_pct)} · {formatPrice(node.price)}</span>
                    <p>{node.reason}</p>
                    <div>
                      {(node.factors || []).slice(0, 3).map((factor) => (
                        <small key={factor}>{factor}</small>
                      ))}
                    </div>
                  </article>
                ))
              ) : (
                <p>当前样本期内没有单日超过 2% 的黄金价格变动。</p>
              )}
            </div>
          </div>
        </div>
      ) : (
        <PlaceholderPanel icon={LineChart} text={loading ? '正在读取真实黄金历史行情。' : '暂无可展示的黄金历史行情。'} />
      )}
    </section>
  );
}

function RiskBudgetPanel({ budget, profile }) {
  return (
    <section className={`risk-budget-panel level-${budget.level}`}>
      <PanelTitle icon={Calculator} title="风险预算计算器" subtitle="以 10 万资金估算本轮风险容量" />
      <div className="budget-grid">
        <MetricLine label="计划黄金暴露" value={formatCurrency(budget.plannedExposure)} />
        <MetricLine label="最大可承受亏损" value={formatCurrency(budget.maxLoss)} />
        <MetricLine label="建议暴露上限" value={formatCurrency(budget.suggestedExposure)} />
        <MetricLine label="持仓模式" value={positionLabels[profile.current_position]} />
      </div>
      <p>{budget.guidance}</p>
    </section>
  );
}

function SuitabilityGatePanel({ gate }) {
  return (
    <section className={`suitability-gate-panel level-${gate.level}`}>
      <PanelTitle icon={ShieldCheck} title="适当性门控预检" subtitle={gate.subtitle} />
      <div className="gate-decision-row">
        <strong>{gate.decision}</strong>
        <span>{gate.scoreLabel}</span>
      </div>
      <p>{gate.summary}</p>
      <div className="gate-rule-list">
        {gate.rules.map((rule) => (
          <span key={rule}>{rule}</span>
        ))}
      </div>
    </section>
  );
}

function PositionModePanel({ budget, profile }) {
  return (
    <section className="position-mode-panel">
      <PanelTitle icon={WalletCards} title="持仓模式" subtitle={positionLabels[profile.current_position]} />
      <p>{budget.positionGuidance}</p>
      <div className="position-rule-stack">
        {budget.rules.map((rule) => (
          <span key={rule}>{rule}</span>
        ))}
      </div>
    </section>
  );
}

function ExecutionScenarioPanel({ scenarios }) {
  return (
    <section className="execution-scenario-panel">
      <SectionHeader
        kicker="Execution Scenarios"
        title="三情景执行框架"
        description="把 Agent 结论转成可观察、可停止、可复盘的执行边界。"
      />
      <div className="execution-scenario-grid">
        {scenarios.map((scenario) => (
          <article key={scenario.name} className={`execution-card ${scenario.tone}`}>
            <span>{scenario.name}</span>
            <strong>{scenario.action}</strong>
            <p>{scenario.condition}</p>
            <small>{scenario.stop}</small>
          </article>
        ))}
      </div>
    </section>
  );
}

function ForecastCard({ forecast }) {
  return (
    <article className={`forecast-card ${stanceClass[forecast.stance] || 'tone-neutral'}`}>
      <div>
        <span>{horizonShortLabels[forecast.horizon] || forecast.horizon}</span>
        <small>{forecastBasisLabel(forecast)}</small>
      </div>
      <strong>{forecast.stance}</strong>
      <p>{forecast.action} · {forecast.confidence_band}置信度 · {(forecast.probability * 100).toFixed(1)}%</p>
      <ul>
        {(forecast.reasons || []).slice(0, 3).map((reason) => (
          <li key={reason}>{reason}</li>
        ))}
      </ul>
    </article>
  );
}

function IndicatorGroupCard({ group }) {
  const Icon = groupIcons[group.id] || Gauge;
  const [openIndicatorId, setOpenIndicatorId] = useState(null);
  return (
    <article className={`indicator-card status-${group.status}`}>
      <div className="indicator-card-head">
        <span>
          <Icon size={16} />
          {group.title}
        </span>
        <QualityDot status={group.status} />
      </div>
      <p>{group.summary}</p>
      <div className="score-line">
        <span>score {group.score.toFixed(2)}</span>
        <span>{group.freshness_seconds}s</span>
      </div>
      <div className="indicator-list">
        {group.indicators.map((indicator) => {
          const isOpen = openIndicatorId === indicator.id;
          return (
            <div key={indicator.id} className={`indicator-audit-shell direction-${indicator.direction}`}>
              <button
                type="button"
                className="indicator-row"
                aria-label={`审计 ${indicator.label}`}
                aria-expanded={isOpen}
                aria-controls={`indicator-audit-${indicator.id}`}
                onClick={() => setOpenIndicatorId(isOpen ? null : indicator.id)}
              >
                <div>
                  <strong>{indicator.label}</strong>
                  <span>{indicator.source}</span>
                </div>
                <div>
                  <strong>{indicator.value}</strong>
                  <span>{indicatorDirectionLabel[indicator.direction] || indicator.direction}</span>
                </div>
                <span className="audit-trigger">
                  <ChevronRight size={14} className={isOpen ? 'open' : ''} />
                  审计
                </span>
              </button>
              {isOpen ? <IndicatorAuditDetails group={group} indicator={indicator} /> : null}
            </div>
          );
        })}
      </div>
    </article>
  );
}

function IndicatorAuditDetails({ group, indicator }) {
  const status = indicator.status || group.status || 'unknown';
  const freshness = indicator.freshness_seconds ?? group.freshness_seconds;
  const degradedReason = indicator.degraded_reason || group.degraded_reason || '无';

  return (
    <section id={`indicator-audit-${indicator.id}`} className="indicator-audit-drawer">
      <div className="audit-drawer-head">
        <div>
          <span>{group.title}</span>
          <h3>指标证据审计</h3>
        </div>
        <QualityDot status={status} />
      </div>
      <div className="audit-fact-grid">
        <span>来源 {indicator.source || 'N/A'}</span>
        <span>状态 {status}</span>
        <span>新鲜度 {freshness != null ? `${freshness}s` : 'N/A'}</span>
        <span>降级原因 {degradedReason}</span>
        <span>研究口径 指标用于解释市场基线，不直接改写预测价格。</span>
        <span>数值口径 {indicator.unit ? `${indicator.numeric_value ?? 'N/A'} ${indicator.unit}` : indicator.numeric_value ?? indicator.value ?? 'N/A'}</span>
      </div>
      {indicator.source_url ? (
        <a className="audit-source-link" href={indicator.source_url} target="_blank" rel="noreferrer">
          打开原始来源
          <ExternalLink size={13} />
        </a>
      ) : (
        <p className="muted-copy">该指标当前使用内部快照或代理数据，暂无外部链接。</p>
      )}
    </section>
  );
}

function QualityPanel({ quality, flags }) {
  return (
    <section className="panel-block">
      <PanelTitle icon={DatabaseZap} title="数据质量" subtitle={quality?.status || 'unknown'} />
      <div className="quality-list">
        <MetricLine label="指标状态" value={quality?.indicator_status || 'N/A'} />
        <MetricLine label="新闻状态" value={quality?.news_status || 'N/A'} />
        <MetricLine label="新鲜度" value={quality ? `${quality.freshness_seconds}s` : 'N/A'} />
      </div>
      {flags.length ? (
        <div className="flag-stack">
          {flags.slice(0, 5).map((flag) => (
            <span key={flag}>{flag}</span>
          ))}
        </div>
      ) : (
        <p className="muted-copy">暂无降级标记。</p>
      )}
    </section>
  );
}

function SourceHealthPanel({ sources, loading }) {
  return (
    <section className="source-health-panel">
      <PanelTitle icon={LockKeyhole} title="数据源健康监控" subtitle={sources.length ? `${sources.length} 个来源` : '等待'} />
      <div className="source-health-list">
        {sources.length ? (
          sources.map((source) => (
            <article key={source.id} className={`source-health-row status-${source.status}`}>
              <div className="source-health-topline">
                <strong>{source.label}</strong>
                <QualityDot status={source.status} />
              </div>
              <div className="source-health-meta">
                <span>{source.cadence}</span>
                <span>{source.freshness_seconds}s / SLA {source.expected_lag_seconds}s</span>
              </div>
              <div className="source-coverage">
                {(source.coverage || []).slice(0, 3).map((item) => (
                  <span key={item}>{item}</span>
                ))}
              </div>
              {source.degraded_reason ? <p>{source.degraded_reason}</p> : <p>来源状态正常，暂无降级原因。</p>}
              {source.url ? (
                <a href={source.url} target="_blank" rel="noreferrer">
                  来源入口
                  <ExternalLink size={13} />
                </a>
              ) : null}
            </article>
          ))
        ) : (
          <p>{loading ? '正在读取数据源健康状态。' : 'dashboard 暂未返回 source_health。'}</p>
        )}
      </div>
    </section>
  );
}

function NewsPanel({ news, loading }) {
  return (
    <section className="panel-block">
      <PanelTitle icon={Newspaper} title="自动情报" subtitle={news.length ? `${news.length} 条` : '等待'} />
      <div className="compact-list">
        {news.length ? (
          news.slice(0, 3).map((item) => (
            <article key={item.event_id}>
              <span>{item.source} · {formatDateTime(item.published_at)}</span>
              <strong>{item.title}</strong>
              <p>{item.summary}</p>
            </article>
          ))
        ) : (
          <p>{loading ? '正在读取近端新闻。' : '暂无新闻条目。'}</p>
        )}
      </div>
    </section>
  );
}

function CitationPanel({ citations }) {
  return (
    <section className="panel-block">
      <PanelTitle icon={FileSearch} title="指标引用" subtitle={`${citations.length} 条`} />
      <div className="compact-list">
        {citations.length ? (
          citations.slice(0, 4).map((item) => (
            <article key={item.id}>
              <span>{item.source_type}</span>
              <strong>{item.label}</strong>
              <p>{item.excerpt}</p>
              {item.url ? (
                <a href={item.url} target="_blank" rel="noreferrer">
                  来源
                  <ExternalLink size={13} />
                </a>
              ) : null}
            </article>
          ))
        ) : (
          <p>指标接口返回后会展示引用。</p>
        )}
      </div>
    </section>
  );
}

function TerminalFooter({ left, right }) {
  return (
    <footer className="terminal-footer">
      <span>{left}</span>
      <span>{right}</span>
    </footer>
  );
}

function SectionHeader({ kicker, title, description }) {
  return (
    <div className="section-head">
      <div>
        <span>{kicker}</span>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
    </div>
  );
}

function PanelTitle({ icon: Icon, title, subtitle }) {
  return (
    <div className="panel-title">
      <Icon size={16} />
      <div>
        <h2>{title}</h2>
        <span>{subtitle}</span>
      </div>
    </div>
  );
}

function MetricTile({ label, value, detail, tone, icon: Icon }) {
  return (
    <article className={`metric-tile tone-${tone}`}>
      <span>
        <Icon size={16} />
        {label}
      </span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function MetricLine({ label, value }) {
  return (
    <div className="metric-line">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatusBadge({ status, loading }) {
  const Icon = loading ? Loader2 : status === 'error' ? AlertTriangle : status === 'degraded' ? AlertTriangle : CheckCircle2;
  return (
    <div className={`status-badge status-${status}`}>
      <Icon size={16} className={loading ? 'spinning' : ''} />
      {loading ? '读取中' : status === 'error' ? '接口异常' : status === 'degraded' ? '含降级' : '已就绪'}
    </div>
  );
}

function QualityDot({ status }) {
  return (
    <span className={`quality-dot status-${status}`}>
      {status}
    </span>
  );
}

function ErrorPanel({ title, message }) {
  return (
    <div className="error-panel">
      <AlertTriangle size={18} />
      <div>
        <strong>{title}</strong>
        <p>{message}</p>
      </div>
    </div>
  );
}

function PlaceholderPanel({ icon: Icon, text, spinning = false }) {
  return (
    <div className="placeholder-panel">
      <Icon size={20} className={spinning ? 'spinning' : ''} />
      <span>{text}</span>
    </div>
  );
}

function SegmentedControl({ label, value, options, onChange }) {
  return (
    <div className="segmented-block">
      <span>{label}</span>
      <div className="segmented-control">
        {options.map(([optionValue, optionLabel]) => (
          <button
            key={optionValue}
            type="button"
            className={value === optionValue ? 'active' : ''}
            onClick={() => onChange(optionValue)}
          >
            {optionLabel}
          </button>
        ))}
      </div>
    </div>
  );
}

function SelectField({ label, value, options, onChange }) {
  return (
    <label className="field compact-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map(([optionValue, optionLabel]) => (
          <option key={optionValue} value={optionValue}>
            {optionLabel}
          </option>
        ))}
      </select>
    </label>
  );
}

function NumberField({ label, suffix, value, min, max, onChange }) {
  return (
    <label className="field compact-field">
      <span>{label}</span>
      <div className="number-input">
        <input
          type="number"
          min={min}
          max={max}
          value={value}
          onChange={(event) => onChange(event.target.value)}
        />
        <small>{suffix}</small>
      </div>
    </label>
  );
}

function SummaryList({ title, items }) {
  return (
    <div className="summary-list">
      <h3>{title}</h3>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

function buildDashboardInsights({ market, forecasts, groups, news, dataQuality, degradationFlags }) {
  const primary = forecasts?.[0];
  const drivers = (groups || []).map((group) => {
    const score = Number(group.score || 0);
    const tone = score > 0.08 ? 'supportive' : score < -0.08 ? 'pressure' : group.status === 'degraded' ? 'watch' : 'neutral';
    return {
      id: group.id,
      title: group.title,
      score: Math.min(1, Math.abs(score)),
      scoreLabel: `${score >= 0 ? '+' : ''}${score.toFixed(2)}`,
      tone,
      summary: group.status === 'degraded' ? '含代理/降级数据，降低权重' : group.summary,
    };
  });
  const supportive = [...drivers].sort((a, b) => Number.parseFloat(b.scoreLabel) - Number.parseFloat(a.scoreLabel))[0];
  const pressure = [...drivers].sort((a, b) => Number.parseFloat(a.scoreLabel) - Number.parseFloat(b.scoreLabel))[0];
  const degraded = degradationFlags?.length || dataQuality?.status === 'degraded' || market?.is_stale;
  const headline = primary
    ? `${horizonShortLabels[primary.horizon]} ${primary.stance}，执行倾向：${primary.action}`
    : '等待稳定预测基线';
  const thesis = {
    kicker: degraded ? '含降级数据，先看风险' : '稳定基线已就绪',
    headline,
    summary: primary
      ? `当前核心不是追逐单点涨跌，而是围绕 ${primary.confidence_band} 置信度管理节奏。${primary.reasons?.[0] || ''}`
      : '读取三周期预测后，这里会收敛为今日核心判断。',
    support: supportive?.title || '等待指标',
    pressure: pressure?.scoreLabel?.startsWith('-') ? pressure.title : degraded ? '数据质量' : '暂无明显压制',
    invalidator: degraded ? '数据恢复前不放大仓位' : '美元与实际利率同步走强',
    tone: primary ? toneName(primary.stance) : 'neutral',
  };

  const newsTitle = news?.[0]?.title || '近端新闻刷新';
  const events = [
    {
      window: 'T-24h',
      name: 'CPI / PCE 通胀数据',
      watch: '若通胀高于预期，实际利率与美元可能压制黄金。',
      impact: 'high',
      impactLabel: '高敏感',
    },
    {
      window: 'T-48h',
      name: 'FOMC / Fed 讲话',
      watch: '关注降息路径、点阵图和鹰鸽措辞变化。',
      impact: 'high',
      impactLabel: '政策敏感',
    },
    {
      window: 'Weekly',
      name: 'ETF / CFTC 资金流',
      watch: '如果价格上行但资金不跟随，趋势可信度下降。',
      impact: 'medium',
      impactLabel: '资金确认',
    },
    {
      window: 'Live',
      name: newsTitle,
      watch: '新闻冲击只作为解释层，不改写稳定预测基线。',
      impact: 'medium',
      impactLabel: '情绪扰动',
    },
  ];

  const primaryPath = primary ? `${primary.action}，不突破风险预算` : '等待预测';
  const scenarios = [
    {
      name: '基准情景',
      path: primaryPath,
      trigger: primary?.reasons?.[0] || '市场基线维持当前方向。',
      invalidator: degraded ? '数据质量恢复前不扩大结论' : '证据冲突扩大则降级',
      tone: 'base',
    },
    {
      name: '偏多情景',
      path: '分批观察上行延续',
      trigger: '美元走弱、实际利率回落、ETF 或避险资金确认。',
      invalidator: '冲高后资金不跟随，回到观望。',
      tone: 'bull',
    },
    {
      name: '偏空情景',
      path: '降低暴露或等待回撤',
      trigger: '美元与实际利率同步上行，技术面跌破关键区间。',
      invalidator: '避险需求重新抬升且金价收复关键位。',
      tone: 'bear',
    },
  ];
  return { thesis, drivers, events, scenarios };
}

function buildRiskBudget(profile, profileScore) {
  const allocation = Number(profile.capital_allocation_pct || 0);
  const maxDrawdown = Number(profile.max_drawdown_pct || 0);
  const plannedExposure = baseCapital * allocation / 100;
  const maxLoss = baseCapital * maxDrawdown / 100;
  const leverageHaircut = { none: 1, low: 0.75, medium: 0.5, high: 0.25 }[profile.leverage_attitude] || 1;
  const scoreHaircut = profileScore >= 5 ? 0.35 : profileScore >= 2 ? 0.65 : 1;
  const liquidityHaircut = profile.liquidity_need === 'high' ? 0.6 : profile.liquidity_need === 'medium' ? 0.85 : 1;
  const suggestedExposure = plannedExposure * leverageHaircut * scoreHaircut * liquidityHaircut;
  const level = profileScore >= 5 ? 'high' : profileScore >= 2 ? 'medium' : 'low';
  const positionGuidance = {
    none: '无持仓时先看触发条件，不用一次性把风险预算打满。',
    long: '已有多头时先管理现有仓位，新增暴露必须等待确认信号。',
    short: '已有空头时优先检查偏多失效条件，避免与基线方向硬扛。',
    hedged: '已对冲时重点观察对冲是否过度，不急于拆腿。',
  }[profile.current_position];
  const rules = [
    allocation >= 50 ? '禁止重仓追价' : '分批进入',
    maxDrawdown <= 5 ? '回撤触线即停止' : '按失效条件复盘',
    profile.leverage_attitude === 'high' ? '不建议使用高杠杆' : '不放大杠杆',
  ];
  const guidance = level === 'high'
    ? '问卷风险偏高，建议把实际暴露压到计划值的一小部分，优先等待确认。'
    : level === 'medium'
      ? '风险预算可以使用，但需要分批和明确失效条件。'
      : '问卷风险较低，可以把重点放在触发条件和复盘纪律。';
  return {
    plannedExposure,
    maxLoss,
    suggestedExposure,
    level,
    guidance,
    positionGuidance,
    rules,
  };
}

function buildSuitabilityGate(profile, profileScore, riskBudget) {
  const allocation = Number(profile.capital_allocation_pct || 0);
  const maxDrawdown = Number(profile.max_drawdown_pct || 0);
  const rules = [];

  if (allocation >= 50) rules.push('禁止重仓追价');
  if (maxDrawdown <= 5) rules.push('回撤触线即停止');
  if (profile.leverage_attitude === 'high') rules.push('禁止加杠杆');
  if (profile.experience_level === 'beginner') rules.push('新手账户只看确认信号');
  if (profile.current_position === 'long') rules.push('已有多头先管存量');
  if (profile.current_position === 'short') rules.push('已有空头先看偏多失效');
  if (profile.liquidity_need === 'high') rules.push('保留流动性缓冲');
  if (!rules.length) rules.push('允许继续分析但不自动执行');

  const forceObservation =
    profileScore >= 7 ||
    (allocation >= 50 && maxDrawdown <= 5) ||
    (profile.leverage_attitude === 'high' && profile.experience_level === 'beginner');
  const level = forceObservation ? 'high' : profileScore >= 3 ? 'medium' : 'low';
  const decision = forceObservation ? '强制观望' : level === 'medium' ? '降低暴露' : '可继续分析';
  const subtitle = level === 'high' ? '高风险门控' : level === 'medium' ? '中风险限制' : '低风险预检';
  const summary = forceObservation
    ? `当前问卷组合超过执行边界，Agent 只能输出观察、失效条件和复盘线索，建议暴露上限 ${formatCurrency(riskBudget.suggestedExposure)}。`
    : level === 'medium'
      ? `当前风险预算需要折扣使用，建议暴露上限 ${formatCurrency(riskBudget.suggestedExposure)}，不得放大仓位。`
      : `当前画像未触发强门控，但仍需等待证据确认，建议暴露上限 ${formatCurrency(riskBudget.suggestedExposure)}。`;

  return {
    level,
    decision,
    subtitle,
    summary,
    scoreLabel: `风险分 ${profileScore}`,
    rules,
  };
}

function buildTrendChart(points, keyNodes) {
  const width = 920;
  const height = 280;
  const padX = 34;
  const padY = 24;
  const prices = points.map((point) => Number(point.price)).filter((value) => Number.isFinite(value));
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const span = Math.max(1, maxPrice - minPrice);
  const lastIndex = Math.max(1, points.length - 1);

  const coords = points.map((point, index) => {
    const x = padX + (index / lastIndex) * (width - padX * 2);
    const y = height - padY - ((Number(point.price) - minPrice) / span) * (height - padY * 2);
    return { ...point, x, y };
  });

  const pathD = coords.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(' ');
  const areaD = `${pathD} L ${coords[coords.length - 1]?.x?.toFixed(2) || padX} ${height - padY} L ${padX} ${height - padY} Z`;
  const coordByDate = new Map(coords.map((point) => [point.date, point]));
  const keyMarkers = (keyNodes || [])
    .map((node) => {
      const coord = coordByDate.get(node.date);
      if (!coord) return null;
      return { ...node, x: coord.x, y: coord.y };
    })
    .filter(Boolean);

  return {
    pathD,
    areaD,
    keyMarkers,
    latestPrice: coords[coords.length - 1]?.price,
  };
}

function buildExecutionScenarios({ summary, selectedForecast, riskBudget, investorProfile }) {
  const action = summary?.action || selectedForecast?.action || '观望';
  const exposureText = formatCurrency(riskBudget.suggestedExposure);
  return [
    {
      name: '基准执行',
      action: action === '观望' ? '等待确认' : `${action}，上限 ${exposureText}`,
      condition: summary?.reasons?.[0] || selectedForecast?.reasons?.[0] || '稳定预测维持当前方向。',
      stop: '若触发任一失效条件，停止新增暴露。',
      tone: 'base',
    },
    {
      name: '偏多突破',
      action: investorProfile.current_position === 'long' ? '只允许小幅加仓' : '先小仓试探',
      condition: '美元走弱、实际利率回落、资金流确认后再行动。',
      stop: '突破后无法站稳或新闻反转，取消追随。',
      tone: 'bull',
    },
    {
      name: '偏空防守',
      action: investorProfile.current_position === 'long' ? '优先减仓' : '保持现金等待',
      condition: '美元和实际利率同步上行，或技术面跌破关键区间。',
      stop: '避险需求重新抬升时重新评估。',
      tone: 'bear',
    },
  ];
}

function toneName(stance) {
  if (stance === '偏多') return 'bull';
  if (stance === '偏空') return 'bear';
  if (stance === '高风险观望') return 'risk';
  return 'neutral';
}

function formatCurrency(value) {
  if (!Number.isFinite(Number(value))) return 'N/A';
  return Number(value).toLocaleString('zh-CN', {
    style: 'currency',
    currency: 'CNY',
    maximumFractionDigits: 0,
  });
}

function formatPrice(value) {
  if (value === null || value === undefined) return 'N/A';
  return Number(value).toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  });
}

function formatPercent(value) {
  if (value === null || value === undefined) return 'N/A';
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatDateTime(value) {
  try {
    return new Date(value).toLocaleString('zh-CN', { hour12: false });
  } catch (error) {
    return value || '未知时间';
  }
}

export default App;
