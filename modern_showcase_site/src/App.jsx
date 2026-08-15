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

import AllocationResearchPanel from './AdvisorPage';
import EventAlertBanner from './EventAlertBanner';
import GlobalSearch from './GlobalSearch';
import ProfileStepper from './ProfileStepper';
import QuantPage from './QuantPage';
import SignalsPage from './SignalsPage';
import { ActiveCaseRibbon, ResearchCaseWorkspace } from './ResearchCasePanel';
import { MarketSnapshotHero, PrimarySourceFeed, ResearchSummary } from './DashboardExperience';
import { AdvancedProfileFields, CoreProfileFields } from './profileFields';
import { getProfileCompletion, loadProfile, saveProfile, toLegacyAnalyzeProfile } from './profileStore';
import { deriveDashboardState } from './presentationPolicy';
import { addSearchEntries } from './searchIndex';
import { ViewModeContext, loadViewMode, saveViewMode, useViewMode } from './viewMode';

const API_URL = import.meta.env.VITE_AGENT_API_URL || '/api/v1/agent/analyze';
const DASHBOARD_URL =
  import.meta.env.VITE_AGENT_DASHBOARD_URL || API_URL.replace('/analyze', '/dashboard/current');
const FEEDBACK_URL = import.meta.env.VITE_AGENT_FEEDBACK_URL || API_URL.replace('/analyze', '/feedback');
const API_KEY = import.meta.env.VITE_AGENT_API_KEY || 'dev-public-key';

// Public research periods. Legacy T+ model keys never reach the UI.
const horizonLabels = {
  short_term: '短期',
  mid_term: '中期',
  long_term: '长期',
};

const horizonSubLabels = {
  short_term: '1–21天',
  mid_term: '1–6月',
  long_term: '6月以上',
};

const horizonShortLabels = horizonLabels;

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

const starterPrompts = [
  '如果今晚 CPI 高于预期，黄金短期应该如何控制风险？',
  '美元指数继续走强时，黄金中期观点的失效条件是什么？',
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
          <Route path="/quant" element={<QuantPage />} />
          <Route path="/advisor" element={<AdvisorWorkbench />} />
          {/* /agent kept as an alias so old links & the unified workbench coincide */}
          <Route path="/agent" element={<AdvisorWorkbench />} />
          <Route path="/signals" element={<SignalsPage />} />
          <Route path="*" element={<DashboardPage />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

function AppShell({ children }) {
  const [mode, setModeState] = useState(loadViewMode);

  function setMode(next) {
    setModeState(next);
    saveViewMode(next);
  }

  useEffect(() => {
    addSearchEntries('pages', [
      { id: 'page-dashboard', source: 'pages', title: '研究主页', hint: '短中长期预测、指标证据与近端新闻', route: '/', keywords: ['dashboard', '主页', '预测', '指标'] },
      { id: 'page-quant', source: 'pages', title: '量化研究面板', hint: 'HMM 状态机、公允价值、情景锥与校准记分卡', route: '/quant', keywords: ['quant', '量化', 'hmm', '校准', '因子'] },
      { id: 'page-signals', source: 'pages', title: '观点书与信号台账', hint: '短中长三尺度观点、每周不可变发布与前向记分卡', route: '/signals', keywords: ['signals', '观点书', '台账', '信号', 'ledger'] },
      { id: 'page-advisor', source: 'pages', title: '个性化投研', hint: '一份画像两种能力：配置研究 + 提问分析', route: '/advisor', keywords: ['advisor', 'agent', '个性化', '画像', '参考区间', '问答', '分析'] },
    ]);
    addSearchEntries(
      'prompts',
      starterPrompts.map((prompt, index) => ({
        id: `prompt-${index}`,
        source: 'prompts',
        title: prompt,
        hint: '在个性化投研的提问分析中使用该问题',
        route: '/advisor',
        keywords: ['提问', '模板'],
      })),
    );
  }, []);

  return (
    <div className={`terminal-shell mode-${mode}`}>
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
          <NavLink to="/quant">
            <Radar size={16} />
            量化研究
          </NavLink>
          <NavLink to="/signals">
            <BookOpenCheck size={16} />
            观点书与台账
          </NavLink>
          <NavLink to="/advisor">
            <WalletCards size={16} />
            个性化投研
          </NavLink>
        </nav>

        <div className="topbar-right">
          <div className="view-mode-toggle" role="group" aria-label="视图模式">
            <button
              type="button"
              className={mode === 'simple' ? 'active' : ''}
              onClick={() => setMode('simple')}
            >
              简明
            </button>
            <button
              type="button"
              className={mode === 'pro' ? 'active' : ''}
              onClick={() => setMode('pro')}
            >
              专业
            </button>
          </div>
          <GlobalSearch />
          <div className="compliance-pill">
            <ShieldCheck size={15} />
            研究辅助 · 非下单系统
          </div>
        </div>
      </header>
      <nav className="mobile-nav" aria-label="移动端主导航">
        <NavLink to="/" end aria-label="研究主页"><LineChart size={18} /><span>市场</span></NavLink>
        <NavLink to="/quant" aria-label="量化研究"><Radar size={18} /><span>模型</span></NavLink>
        <NavLink to="/signals" aria-label="观点书与台账"><BookOpenCheck size={18} /><span>验证</span></NavLink>
        <NavLink to="/advisor" aria-label="个性化投研"><WalletCards size={18} /><span>我的</span></NavLink>
      </nav>
      <EventAlertBanner />
      <ViewModeContext.Provider value={{ mode, setMode }}>
        {children}
      </ViewModeContext.Provider>
    </div>
  );
}

function DashboardPage() {
  const { mode } = useViewMode();
  const isPro = mode === 'pro';
  const [dashboard, setDashboard] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [reloadKey, setReloadKey] = useState(0);

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
  }, [reloadKey]);

  const market = dashboard?.market_status;
  const forecasts = dashboard?.horizon_forecasts || [];
  const groups = dashboard?.indicator_groups || [];
  const dataQuality = dashboard?.data_quality;
  const degradationFlags = dashboard?.degradation_flags || [];
  const news = dashboard?.recent_news || [];
  const citations = dashboard?.citations || [];
  const sourceHealth = dashboard?.source_health || [];
  const goldHistory = dashboard?.gold_history;

  useEffect(() => {
    addSearchEntries(
      'news',
      (dashboard?.recent_news || []).map((item, index) => ({
        id: `news-${index}`,
        source: 'news',
        title: item.title || item.summary || '未命名新闻',
        hint: [item.source, item.published_at || item.published].filter(Boolean).join(' · '),
        route: '/',
        hash: 'panel-news',
        href: item.url || undefined,
        keywords: ['新闻', 'news'],
      })),
    );
  }, [dashboard]);

  const dashboardState = deriveDashboardState({ loading, error, dashboard });
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
        <StatusBadge status={dashboardState.kind === 'unavailable' ? 'error' : dashboardState.kind} loading={loading} />
      </section>

      <MarketSnapshotHero dashboard={dashboard} state={dashboardState} onRetry={() => setReloadKey((value) => value + 1)} />
      <PrimarySourceFeed news={news} state={dashboardState} />
      <ResearchSummary insights={dashboardInsights} state={dashboardState} />

      <section data-dashboard-section="research-case" className="research-case-stage">
        <ResearchCaseWorkspace />
      </section>

      {dashboardState.kind !== 'unavailable' ? (
      <>
      <section className="research-brief-grid">
        <CoreThesisPanel thesis={dashboardInsights.thesis} />
        <DriverMatrix groups={groups} drivers={dashboardInsights.drivers} />
      </section>

      <section className="forecast-grid" aria-label="Forecast horizons">
        {forecasts.length ? (
          forecasts.map((forecast) => <ForecastCard key={forecast.horizon} forecast={forecast} />)
        ) : (
          <PlaceholderPanel icon={Clock3} text={loading ? '正在读取短期 / 中期 / 长期预测基线。' : '暂无预测基线。'} />
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
              groups.map((group) => (
                <IndicatorGroupCard key={group.id} group={group} showDetails={isPro} />
              ))
            ) : (
              <PlaceholderPanel icon={Gauge} text={loading ? '正在读取指标柱。' : '暂无指标数据。'} />
            )}
          </div>
          {!isPro ? (
            <p className="muted-copy simple-mode-hint">
              简明模式已折叠逐项指标审计与来源健康监控——右上角切换「专业」查看全部方法论细节。
            </p>
          ) : null}
        </div>

        <aside className="side-rail">
          <div className="historical-reference"><span>历史研究参考</span><MarketViewSummaryPanel /></div>
          <QualityPanel quality={dataQuality} flags={degradationFlags} />
          {isPro ? <SourceHealthPanel sources={sourceHealth} loading={loading} /> : null}
          {isPro ? <CitationPanel citations={citations} /> : null}
          <Link className="agent-entry" to="/advisor">
            <span>
              <strong>生成个性化研究分析</strong>
              <small>按你的画像输出参考区间、差距与风险提示</small>
            </span>
            <ArrowRight size={17} />
          </Link>
          <Link className="agent-entry" to="/agent">
            <span>
              <strong>进入风险画像 Agent</strong>
              <small>填写完整问卷后生成风险适配 briefing</small>
            </span>
            <ArrowRight size={17} />
          </Link>
        </aside>
      </section>
      </>
      ) : (
        <section className="historical-reference standalone">
          <span>历史研究参考 · 非当前判断</span>
          <MarketViewSummaryPanel />
        </section>
      )}
      <TerminalFooter
        left="GoldenSense Research Dashboard"
        right="价格、指标、来源健康与风险提示统一在首页收口"
      />
    </main>
  );
}

function AdvisorWorkbench() {
  // One shared profile powers both capabilities; the workbench owns it.
  const [profile, setProfile] = useState(loadProfile);
  const [tab, setTab] = useState('allocation'); // 'allocation' | 'qa'
  const [step, setStep] = useState(1);

  useEffect(() => {
    saveProfile(profile);
  }, [profile]);

  const investorProfile = useMemo(() => toLegacyAnalyzeProfile(profile), [profile]);
  const profileCompletion = useMemo(() => getProfileCompletion(profile), [profile]);
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

  function updateProfile(key, value) {
    setProfile((current) => ({ ...current, [key]: value }));
  }

  return (
    <main className="page-surface advisor-page">
      <section className="terminal-header">
        <div>
          <p className="eyebrow">Personalized Research & Q&A</p>
          <h1>个性化投研</h1>
          <p>
            一份画像，两种能力：给出研究口径的<strong>参考配置区间</strong>，或就任意问题生成
            <strong>可追溯的风险适配分析</strong>。画像只存本机浏览器、随请求发送，服务端不存储；输出永不构成买卖指令。
          </p>
        </div>
        {profileCompletion.suitabilityComplete ? (
          <div className={`profile-score level-${profileLevel === '高' ? 'high' : profileLevel === '中' ? 'medium' : 'low'}`}>
            <span>画像风险</span>
            <strong>{profileLevel}</strong>
            <small>score {profileScore}</small>
          </div>
        ) : (
          <div className="profile-score incomplete">
            <span>画像未完成</span>
            <strong>待补全</strong>
            <small>暂不计算个人风险等级</small>
          </div>
        )}
      </section>

      <ActiveCaseRibbon />

      <ProfileStepper
        step={step}
        onStepChange={setStep}
        completion={profileCompletion}
        core={<section className="profile-card">
        <PanelTitle
          icon={SlidersHorizontal}
          title="第一步：核心画像"
          subtitle="先说明你的期限、经验和当前黄金占比"
        />
        <div className="profile-fields">
          <CoreProfileFields profile={profile} onChange={updateProfile} />
        </div>
      </section>}
        suitability={<section className="profile-card">
          <PanelTitle
            icon={ShieldCheck}
            title="第二步：适当性边界"
            subtitle={`还有 ${profileCompletion.suitabilityMissing.length} 项待补全 · 全部只存本机`}
          />
          <div className="profile-fields">
            <AdvancedProfileFields profile={profile} onChange={updateProfile} defaultOpen />
          </div>
        </section>}
        research={<section className="profile-research-step">
        <div className="capability-tabs" role="tablist" aria-label="能力切换">
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'allocation'}
          className={tab === 'allocation' ? 'active' : ''}
          onClick={() => setTab('allocation')}
        >
          <WalletCards size={16} />
          <span>配置研究</span>
          <small>参考区间 · 仓位差距 · 风险提示</small>
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'qa'}
          className={tab === 'qa' ? 'active' : ''}
          onClick={() => setTab('qa')}
        >
          <BrainCircuit size={16} />
          <span>提问分析</span>
          <small>自由提问 · 证据卡 · 失效条件</small>
        </button>
      </div>

      {tab === 'allocation' ? (
        <AllocationResearchPanel profile={profile} />
      ) : (
        <QaAnalysisPanel
          profile={profile}
          investorProfile={investorProfile}
          profileScore={profileScore}
          profileComplete={profileCompletion.suitabilityComplete}
        />
      )}
      </section>}
      />

      <TerminalFooter
        left="GoldenSense 个性化投研"
        right="画像不落库 · 数字来自规则引擎 · 语言经双重把关 · 非投资建议"
      />
    </main>
  );
}

function QaAnalysisPanel({ profile, investorProfile, profileScore, profileComplete }) {
  const [question, setQuestion] = useState(starterPrompts[0]);
  const [horizon, setHorizon] = useState('short_term');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const resultRef = useRef(null);

  const riskProfile = profile.risk_tolerance;
  const summary = analysis?.summary_card;
  const riskBanner = analysis?.risk_banner;
  const forecasts = analysis?.horizon_forecasts || [];
  const evidenceCards = analysis?.evidence_cards || [];
  const citations = analysis?.citations || [];
  const recentNews = analysis?.recent_news || [];
  const selectedForecast = forecasts.find((item) => item.horizon === horizon) || forecasts[0];

  const riskBudget = useMemo(
    () => buildRiskBudget(investorProfile, profileScore, profileComplete),
    [investorProfile, profileScore, profileComplete],
  );
  const suitabilityGate = useMemo(
    () => buildSuitabilityGate(investorProfile, profileScore, riskBudget, profileComplete),
    [investorProfile, profileScore, riskBudget, profileComplete],
  );
  const executionScenarios = useMemo(
    () => buildExecutionScenarios({ summary, selectedForecast, riskBudget, investorProfile }),
    [summary, selectedForecast, riskBudget, investorProfile],
  );

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
      const json = await readApiJson(response, '分析失败');
      setAnalysis(json);
      window.setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    } catch (submitError) {
      setError(submitError.message || '分析失败');
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
      setFeedbackStatus(rating === 'helpful' ? '已记录：这条分析有帮助。' : '已记录：这类回答会进入后续评估。');
    } catch (feedbackError) {
      setFeedbackStatus(feedbackError.message || '反馈提交失败');
    }
  }

  return (
    <div className="capability-panel">
      <div className="capability-intro">
        <p>
          就任意黄金问题生成<strong>风险适配分析</strong>：方向观点、三情景执行框架、失效条件与禁止执行条件，
          全部可回溯到证据卡与引用。市场基线来自行情与量化服务，<strong>不被你的问题改写</strong>。
        </p>
      </div>

      <section className="qa-layout">
        <form className="agent-form" onSubmit={handleSubmit}>
          <label className="field">
            <span>你的问题</span>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={5}
              placeholder="例：如果 CPI 高于预期，黄金短期和一周视角分别要怎么看？"
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
              label="分析周期"
              value={horizon}
              options={Object.entries(horizonLabels)}
              onChange={setHorizon}
            />
          </div>

          <RiskBudgetPanel budget={riskBudget} profile={investorProfile} />
          <SuitabilityGatePanel gate={suitabilityGate} />

          {error ? <ErrorPanel title="分析失败" message={error} /> : null}

          <button className="primary-action" type="submit" disabled={loading}>
            {loading ? <Loader2 size={17} className="spinning" /> : <BrainCircuit size={17} />}
            {loading ? '生成分析中' : '生成风险适配分析'}
          </button>
        </form>

        <aside className="agent-context">
          <PanelTitle icon={LockKeyhole} title="输出纪律" subtitle="工业级研究终端的边界" />
          <ul className="boundary-list">
            <li>市场基线来自行情和量化服务，不被用户问题改写。</li>
            <li>画像只影响风险适配、输出强度和禁止执行条件。</li>
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
              <strong>暂无结果</strong>
              <p>提交后这里会保留当前周期的稳定预测基线。</p>
            </div>
          )}
          <PositionModePanel budget={riskBudget} profile={investorProfile} />
        </aside>
      </section>

      <section ref={resultRef} className="analysis-output">
        {!analysis && !loading ? (
          <PlaceholderPanel icon={BookOpenCheck} text="提交问题后，这里会展示完整分析、证据卡和引用。" />
        ) : null}

        {loading ? (
          <PlaceholderPanel icon={Loader2} text="正在读取预测基线、新闻、历史类比和画像门控。" spinning />
        ) : null}

        {summary ? (
          <>
            <SectionHeader
              kicker="Analysis Briefing"
              title="风险适配分析"
              description="输出方向、情景、失效条件和禁止执行条件，避免把研究解读误读成下单指令。"
            />
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
                  <span>这条分析是否有帮助？</span>
                  <button type="button" onClick={() => submitFeedback('helpful')}>有帮助</button>
                  <button type="button" onClick={() => submitFeedback('not_helpful')}>没帮助</button>
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
            <ExecutionScenarioPanel scenarios={executionScenarios} />
          </>
        ) : null}

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
    </div>
  );
}

const MARKET_VIEW_URL =
  import.meta.env.VITE_AGENT_MARKET_VIEW_URL || API_URL.replace('/analyze', '/market-view');

const marketViewMeta = {
  short_term: '短期 1-21天',
  mid_term: '中期 1-6月',
  long_term: '长期 6月+',
};

function MarketViewSummaryPanel() {
  const [book, setBook] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetch(MARKET_VIEW_URL, { headers: apiHeaders(), signal: controller.signal })
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then(setBook)
      .catch(() => {
        if (!controller.signal.aborted) setFailed(true);
      });
    return () => controller.abort();
  }, []);

  return (
    <section className="panel-block market-view-summary">
      <PanelTitle
        icon={BookOpenCheck}
        title="大盘观点书"
        subtitle={book?.meta?.data_asof ? `数据截至 ${book.meta.data_asof} · 非实时` : '读取中'}
      />
      {book ? (
        <div className="compact-list">
          {Object.entries(marketViewMeta).map(([key, label]) => {
            const section = book[key];
            return (
              <article key={key}>
                <span>{label}{section?.available ? ` · 置信度${section.confidence}` : ''}</span>
                <p>{section?.available ? section.core_view : `数据降级：${section?.degraded_reason || '不可用'}`}</p>
              </article>
            );
          })}
        </div>
      ) : (
        <p className="muted-copy">{failed ? '观点书暂不可用。' : '正在读取三尺度观点。'}</p>
      )}
      <Link className="audit-source-link" to="/signals">
        查看完整观点书与信号台账
        <ArrowRight size={13} />
      </Link>
    </section>
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
      <PanelTitle icon={Calculator} title="风险预算画像" subtitle="全部以组合占比表达，不假设资金规模" />
      <div className="budget-grid">
        <MetricLine label="当前黄金暴露" value={formatPctPlain(budget.plannedExposurePct)} />
        <MetricLine label="声明回撤承受力" value={formatPctPlain(budget.maxLossPct)} />
        <MetricLine label="研究参考暴露上限" value={formatPctPlain(budget.suggestedExposurePct)} />
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

function IndicatorGroupCard({ group, showDetails = true }) {
  const Icon = groupIcons[group.id] || Gauge;
  const [openIndicatorId, setOpenIndicatorId] = useState(null);
  if (!showDetails) {
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
      </article>
    );
  }
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
      path: '观察上行条件是否得到确认',
      trigger: '美元走弱、实际利率回落、ETF 或避险资金确认。',
      invalidator: '冲高后资金不跟随，回到观望。',
      tone: 'bull',
    },
    {
      name: '偏空情景',
      path: '观察下行情景是否持续',
      trigger: '美元与实际利率同步上行，技术面跌破关键区间。',
      invalidator: '避险需求重新抬升且金价收复关键位。',
      tone: 'bear',
    },
  ];
  return { thesis, drivers, events, scenarios };
}

function buildRiskBudget(profile, profileScore, profileComplete = true) {
  // All figures are % of the user's portfolio: the product never assumes a
  // capital amount, and never phrases exposure as advice.
  const allocation = Number(profile.capital_allocation_pct || 0);
  const maxDrawdown = Number(profile.max_drawdown_pct || 0);
  const leverageHaircut = { none: 1, low: 0.75, medium: 0.5, high: 0.25 }[profile.leverage_attitude] || 1;
  const scoreHaircut = profileComplete ? (profileScore >= 5 ? 0.35 : profileScore >= 2 ? 0.65 : 1) : 0;
  const liquidityHaircut = profile.liquidity_need === 'high' ? 0.6 : profile.liquidity_need === 'medium' ? 0.85 : 1;
  const suggestedExposurePct = allocation * leverageHaircut * scoreHaircut * liquidityHaircut;
  const level = profileComplete ? (profileScore >= 5 ? 'high' : profileScore >= 2 ? 'medium' : 'low') : 'neutral';
  const positionGuidance = {
    none: '无持仓时先看触发条件，研究口径不覆盖一次性打满风险预算的路径。',
    long: '已有多头时先关注存量仓位与失效条件，新增暴露的研究前提是确认信号出现。',
    short: '已有空头时优先检查偏多失效条件，留意与基线方向的冲突。',
    hedged: '已对冲时重点观察对冲是否过度，研究口径不含急拆对冲腿的情形。',
  }[profile.current_position];
  const rules = [
    allocation >= 50 ? '重仓追价超出研究口径' : '分批进入',
    maxDrawdown <= 5 ? '回撤触线即停止' : '按失效条件复盘',
    profile.leverage_attitude === 'high' ? '高杠杆超出研究口径' : '不放大杠杆',
  ];
  const guidance = !profileComplete
    ? '适当性边界尚未补全，暂不计算个人风险等级或暴露上限。'
    : level === 'high'
    ? '画像风险偏高：研究参考口径下，实际暴露显著低于计划值、并等待确认信号，是与该画像一致的路径。'
    : level === 'medium'
      ? '风险预算可用的研究前提：分批、且每一步有明确失效条件。'
      : '画像风险较低：重点可放在触发条件与复盘纪律上。';
  return {
    plannedExposurePct: allocation,
    maxLossPct: maxDrawdown,
    suggestedExposurePct,
    level,
    guidance,
    positionGuidance,
    rules,
  };
}

function buildSuitabilityGate(profile, profileScore, riskBudget, profileComplete = true) {
  const allocation = Number(profile.capital_allocation_pct || 0);
  const maxDrawdown = Number(profile.max_drawdown_pct || 0);
  const rules = [];

  if (!profileComplete) {
    return {
      level: 'neutral',
      decision: '暂不评级',
      subtitle: '适当性信息未完成',
      summary: '请先补全回撤、流动性、工具与法域等边界；当前只能进行一般教育型研究。',
      scoreLabel: '画像未完成',
      rules: ['补全适当性边界后再进行个人风险评级'],
    };
  }

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
  const decision = forceObservation ? '强制观望' : level === 'medium' ? '风险限制' : '可继续分析';
  const subtitle = level === 'high' ? '高风险门控' : level === 'medium' ? '中风险限制' : '低风险预检';
  const exposureText = formatPctPlain(riskBudget.suggestedExposurePct);
  const summary = forceObservation
    ? `当前画像组合超过执行边界，Agent 只能输出观察、失效条件和复盘线索，研究参考暴露上限 ${exposureText}（组合占比）。`
    : level === 'medium'
      ? `当前风险预算需要折扣使用，研究参考暴露上限 ${exposureText}（组合占比），研究口径不含放大仓位的路径。`
      : `当前画像未触发强门控，但仍需等待证据确认，研究参考暴露上限 ${exposureText}（组合占比）。`;

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
  return [
    {
      name: '基准观察',
      action: '观察基础情景是否得到确认',
      condition: summary?.reasons?.[0] || selectedForecast?.reasons?.[0] || '稳定预测维持当前方向。',
      stop: '若触发任一失效条件，基础情景作废。',
      tone: 'base',
    },
    {
      name: '偏多突破',
      action: '观察上行条件是否持续',
      condition: '美元走弱、实际利率回落、资金流确认作为情景成立依据。',
      stop: '突破后无法站稳或新闻反转，上行情景失效。',
      tone: 'bull',
    },
    {
      name: '偏空防守',
      action: '观察下行风险是否持续',
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

function formatPctPlain(value) {
  if (!Number.isFinite(Number(value))) return 'N/A';
  return `${Number(value).toFixed(1)}%`;
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
