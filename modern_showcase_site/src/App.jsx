import React, { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import {
  AlertTriangle,
  ArrowRight,
  BadgeAlert,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  Clock3,
  ExternalLink,
  FileSearch,
  Flame,
  Landmark,
  Loader2,
  MessagesSquare,
  Newspaper,
  Radar,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Waves,
  Zap,
} from 'lucide-react';

const API_URL = import.meta.env.VITE_AGENT_API_URL || '/api/v1/agent/analyze';
const FEEDBACK_URL = import.meta.env.VITE_AGENT_FEEDBACK_URL || API_URL.replace('/analyze', '/feedback');
const API_KEY = import.meta.env.VITE_AGENT_API_KEY || 'dev-public-key';

const starterPrompts = [
  '今晚 CPI 如果高于预期，黄金短线应该怎么看？',
  '我比较保守，现在是不是适合小仓试探黄金？',
  '地缘冲突升级的话，黄金的 T+1、T+7、T+30 应该分别怎么看？',
  '如果美元继续走强，黄金这周更适合观望还是降低暴露？',
];

const horizonLabels = {
  '24h': '24 小时',
  '7d': '7 天',
  '30d': '30 天',
};

const horizonTags = {
  '24h': 'T+1',
  '7d': 'T+7',
  '30d': 'T+30',
};

const riskLabels = {
  conservative: '保守型',
  balanced: '平衡型',
  aggressive: '进取型',
};

const riskDescriptions = {
  conservative: '优先控制回撤，建议更偏小仓和等待确认。',
  balanced: '接受适度波动，更看重风险收益比的平衡。',
  aggressive: '允许更高波动，但依然要看失效条件和风险条。',
};

const horizonDescriptions = {
  '24h': '更偏事件驱动与短线波动，适合看下一次宏观催化会不会改写方向。',
  '7d': '一周视角更适合判断情绪是否延续，以及风险偏好会不会持续切换。',
  '30d': '中期参考更适合看趋势结构和叙事是否真正站稳，而不是单一数据点。',
};

const stanceTone = {
  偏多: 'tone-bull',
  偏空: 'tone-bear',
  中性: 'tone-neutral',
  高风险观望: 'tone-risk',
};

const signalTypeMeta = {
  market: { label: '市场状态', icon: BarChart3 },
  quant: { label: '量化预测', icon: BrainCircuit },
  news: { label: '新闻情绪', icon: Flame },
  memory: { label: '历史类比', icon: FileSearch },
  macro: { label: '宏观语境', icon: Landmark },
  risk: { label: '风险画像', icon: ShieldCheck },
};

const basisLabels = {
  ensemble_model: '模型直推',
  heuristic_proxy: '代理预测',
  degraded_fallback: '降级回退',
};

const riskOptions = Object.entries(riskLabels).map(([value, label]) => ({ value, label }));
const horizonOptions = Object.entries(horizonLabels).map(([value, label]) => ({ value, label }));

function App() {
  const shouldReduceMotion = useReducedMotion();
  const [question, setQuestion] = useState(starterPrompts[0]);
  const [riskProfile, setRiskProfile] = useState('conservative');
  const [horizon, setHorizon] = useState('24h');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [conversation, setConversation] = useState([]);
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [highlightedCitation, setHighlightedCitation] = useState('');
  const [tickerIndex, setTickerIndex] = useState(0);

  const deferredQuestion = useDeferredValue(question);
  const formRef = useRef(null);
  const briefingRef = useRef(null);
  const citationTimeoutRef = useRef(null);

  const summary = analysis?.summary_card;
  const riskBanner = analysis?.risk_banner;
  const evidenceCards = analysis?.evidence_cards || [];
  const citations = analysis?.citations || [];
  const horizonForecasts = analysis?.horizon_forecasts || [];
  const recentNews = analysis?.recent_news || [];
  const followUpQuestions = analysis?.follow_up_questions || [];
  const degradationFlags = analysis?.degradation_flags || [];

  function apiHeaders(extra = {}) {
    return {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
      ...extra,
    };
  }

  const containerVariants = useMemo(
    () => ({
      hidden: { opacity: 1 },
      show: {
        opacity: 1,
        transition: {
          staggerChildren: shouldReduceMotion ? 0 : 0.08,
          delayChildren: shouldReduceMotion ? 0 : 0.02,
        },
      },
    }),
    [shouldReduceMotion],
  );

  const itemVariants = useMemo(
    () => ({
      hidden: { opacity: 0, y: shouldReduceMotion ? 0 : 18 },
      show: {
        opacity: 1,
        y: 0,
        transition: { duration: shouldReduceMotion ? 0.08 : 0.45, ease: 'easeOut' },
      },
    }),
    [shouldReduceMotion],
  );

  const quickMood = useMemo(() => {
    if (loading) {
      return '正在整合市场快照、量化信号、新闻和历史类比。';
    }
    if (!summary) {
      return '把宏观新闻、量化信号和历史类比压缩成散户也能快速读懂的判断。';
    }
    return `${summary.stance} · ${summary.action} · ${summary.confidence_band}置信度`;
  }, [loading, summary]);

  const selectedForecast = useMemo(() => {
    if (!horizonForecasts.length) return null;
    return horizonForecasts.find((item) => item.horizon === horizon) || horizonForecasts[0];
  }, [horizon, horizonForecasts]);

  const analysisStats = useMemo(
    () => [
      {
        label: '新闻',
        value: recentNews.length ? `${recentNews.length} 条` : '0 条',
      },
      {
        label: '证据',
        value: evidenceCards.length ? `${evidenceCards.length} 张` : '0 张',
      },
      {
        label: '引用',
        value: citations.length ? `${citations.length} 条` : '0 条',
      },
      {
        label: '耗时',
        value: analysis?.timing_ms?.total ? `${analysis.timing_ms.total} ms` : '未执行',
      },
    ],
    [analysis, citations.length, evidenceCards.length, recentNews.length],
  );

  const workflowSteps = loading
    ? [
        '读取统一市场快照与风险环境',
        '拉取近端新闻并生成多周期判断',
        '按用户画像收敛为可执行建议',
      ]
    : summary
      ? [
          `主结论：${summary.stance}，建议 ${summary.action}`,
          `重点风险：${riskBanner?.title || '等待风险条'}`,
          '你可以继续追问关键价位、仓位或失效情景',
        ]
      : [
          '输入你的问题，系统会自动抓取近端新闻与市场快照',
          '同时生成 T+1 / T+7 / T+30 三档判断',
          '按散户风险偏好重写成可执行建议',
        ];

  const heroStatus = loading ? 'Agent 正在整理本轮信号' : summary ? '最近一次分析已完成' : '等待你发起问题';
  const heroStatusTone = loading ? 'pending' : summary ? 'ready' : 'idle';

  const heroTickerItems = useMemo(() => {
    if (loading) {
      return [
        '正在读取市场快照、量化信号、新闻和历史类比。',
        '本轮会同时生成 T+1、T+7、T+30 三档判断。',
        '风险偏好与分析周期已经锁定到当前输入里。',
      ];
    }

    if (summary) {
      return [
        quickMood,
        `当前锁定：${riskLabels[riskProfile]} · ${horizonLabels[horizon]}`,
        `已纳入 ${recentNews.length} 条新闻、${evidenceCards.length} 张证据卡、${citations.length} 条引用。`,
        riskBanner?.message || '所有结论都必须能回到证据文本和失效条件。',
      ];
    }

    return [
      '从一个更具体的问题开始，GoldenSense 才能把判断收敛到可执行的下一步。',
      `当前默认：${riskLabels[riskProfile]} · ${horizonLabels[horizon]}。`,
      '你不需要手动贴新闻，系统会自动抓取本轮近端情报。',
      '输入一个事件、期限和立场疑问，首轮结果会更有用。',
    ];
  }, [citations.length, evidenceCards.length, horizon, loading, quickMood, recentNews.length, riskBanner?.message, riskProfile, summary]);

  const heroTickerText = heroTickerItems[tickerIndex] || heroTickerItems[0];

  useEffect(() => {
    setTickerIndex(0);
  }, [heroTickerItems.length, loading, horizon, riskProfile, summary]);

  useEffect(() => {
    if (shouldReduceMotion || heroTickerItems.length <= 1) {
      return undefined;
    }

    const intervalId = window.setInterval(() => {
      setTickerIndex((current) => (current + 1) % heroTickerItems.length);
    }, 2800);

    return () => window.clearInterval(intervalId);
  }, [heroTickerItems, shouldReduceMotion]);

  useEffect(
    () => () => {
      if (citationTimeoutRef.current) {
        window.clearTimeout(citationTimeoutRef.current);
      }
    },
    [],
  );

  async function handleSubmit(event) {
    event.preventDefault();

    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      setError('请输入你的问题，再开始分析。');
      return;
    }

    setLoading(true);
    setError('');
    setFeedbackStatus('');

    const payload = {
      question: trimmedQuestion,
      risk_profile: riskProfile,
      horizon,
      locale: 'zh-CN',
    };

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: apiHeaders(),
        body: JSON.stringify(payload),
      });
      const json = await response.json();
      if (!response.ok) {
        throw new Error(json?.detail?.message || json?.detail || '分析请求失败');
      }

      startTransition(() => {
        setAnalysis(json);
        setConversation((prev) => {
          const next = [
            ...prev,
            {
              id: `${Date.now()}-user`,
              role: 'user',
              content: trimmedQuestion,
            },
            {
              id: `${Date.now()}-assistant`,
              role: 'assistant',
              content: json.summary_card.reasons[0] || `${json.summary_card.stance}，${json.summary_card.action}`,
            },
          ];
          return next.slice(-6);
        });
      });

      window.setTimeout(() => {
        briefingRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 120);
    } catch (submitError) {
      setError(submitError.message || '分析请求失败');
    } finally {
      setLoading(false);
    }
  }

  async function submitFeedback(rating) {
    if (!analysis?.analysis_id || feedbackLoading) return;

    try {
      setFeedbackLoading(true);
      const response = await fetch(FEEDBACK_URL, {
        method: 'POST',
        headers: apiHeaders(),
        body: JSON.stringify({
          analysis_id: analysis.analysis_id,
          rating,
          comment: null,
        }),
      });
      if (!response.ok) {
        throw new Error('反馈提交失败');
      }
      setFeedbackStatus(rating === 'helpful' ? '已记录：这条回答对你有帮助。' : '已记录：我们会重点改进这类回答。');
    } catch (feedbackError) {
      setFeedbackStatus(feedbackError.message || '反馈提交失败');
    } finally {
      setFeedbackLoading(false);
    }
  }

  function handleQuestionKeyDown(event) {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault();
      formRef.current?.requestSubmit();
    }
  }

  function usePrompt(prompt) {
    setQuestion(prompt);
    setError('');
  }

  function handleFollowUp(questionText) {
    setQuestion(questionText);
    setError('');
    formRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function scrollToCitation(citationId) {
    const node = document.getElementById(`citation-${citationId}`);
    if (!node) return;

    node.scrollIntoView({ behavior: 'smooth', block: 'center' });
    setHighlightedCitation(citationId);

    if (citationTimeoutRef.current) {
      window.clearTimeout(citationTimeoutRef.current);
    }
    citationTimeoutRef.current = window.setTimeout(() => setHighlightedCitation(''), 1800);
  }

  function scrollToBriefing() {
    briefingRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  return (
    <div className="consumer-shell">
      <section className="hero-stage">
        <div className="hero-inner">
          <header className="hero-masthead">
            <div className="brand-cluster">
              <div className="brand-mark">
                <Sparkles size={16} />
              </div>
              <div className="brand-copy">
                <span>GoldenSense Retail Agent</span>
                <strong>黄金判断不是一句喊单，而是一份可追溯的 briefing</strong>
              </div>
            </div>
            <div className={`page-status ${heroStatusTone}`}>
              {loading ? <Loader2 size={14} className="spinning" /> : <CheckCircle2 size={14} />}
              {heroStatus}
            </div>
          </header>

          <div className="hero-grid">
            <motion.div className="hero-story" initial="hidden" animate="show" variants={containerVariants}>
              <motion.div variants={itemVariants} className="eyebrow">
                <Sparkles size={14} />
                Education-first Gold Agent
              </motion.div>

              <motion.div variants={itemVariants} className="hero-copy-block">
                <p className="hero-kicker">Retail Gold Briefing Engine</p>
                <h1>现在怎么看黄金</h1>
                <p className="hero-description">
                  面向中文散户的黄金投资助手。它不会替你下单，而是把新闻、量化、历史剧本和风险边界整理成一份可以继续追问的判断。
                </p>
              </motion.div>

              <motion.div variants={itemVariants} className="hero-band">
                <div className={`signal-chip ${summary ? stanceTone[summary.stance] : 'tone-outline'}`}>
                  {summary ? summary.stance : '尚未生成结论'}
                </div>
                <div className="signal-chip tone-outline">风险画像：{riskLabels[riskProfile]}</div>
                <div className="signal-chip tone-outline">主结论视角：{horizonLabels[horizon]}</div>
                {degradationFlags.length ? <div className="signal-chip tone-risk">数据降级：{degradationFlags.length} 项</div> : null}
              </motion.div>

              <motion.div variants={itemVariants} className="hero-ticker-shell">
                <div className="ticker-label">
                  <Radar size={14} />
                  Live Signal
                </div>
                <motion.div
                  key={heroTickerText}
                  className="hero-ticker-current"
                  initial={{ opacity: 0, y: shouldReduceMotion ? 0 : 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: shouldReduceMotion ? 0.08 : 0.28 }}
                >
                  {heroTickerText}
                </motion.div>
              </motion.div>

              <motion.form
                ref={formRef}
                variants={itemVariants}
                className="hero-console"
                onSubmit={handleSubmit}
              >
                <div className="console-topline">
                  <div>
                    <h2>立即发起一轮分析</h2>
                    <p>把问题直接丢给 Agent，系统会自动抓取市场快照、新闻与历史类比，并回到同一张判断卡里。</p>
                  </div>
                  <button className="submit-button hero-submit" type="submit" disabled={loading}>
                    {loading ? '分析中…' : '开始分析'}
                    {loading ? <Loader2 size={16} className="spinning" /> : <ArrowRight size={16} />}
                  </button>
                </div>

                <label className="field console-field">
                  <span>你的问题</span>
                  <textarea
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    onKeyDown={handleQuestionKeyDown}
                    rows={5}
                    placeholder="例：如果本周美联储偏鹰，黄金 24 小时和 7 天分别要怎么看？"
                  />
                </label>

                <div className="selection-strip hero-selection-strip">
                  <SelectionBadge label="风险偏好" value={riskLabels[riskProfile]} />
                  <SelectionBadge label="分析周期" value={horizonLabels[horizon]} />
                  <SelectionBadge label="发送捷径" value="Cmd/Ctrl + Enter" />
                </div>

                <div className="control-grid">
                  <SegmentedControl
                    label="风险偏好"
                    value={riskProfile}
                    options={riskOptions}
                    description={riskDescriptions[riskProfile]}
                    disabled={loading}
                    onChange={setRiskProfile}
                  />
                  <SegmentedControl
                    label="分析周期"
                    value={horizon}
                    options={horizonOptions}
                    description={horizonDescriptions[horizon]}
                    disabled={loading}
                    onChange={setHorizon}
                  />
                </div>

                <div className="prompt-rail">
                  <div className="rail-label">
                    <Zap size={14} />
                    推荐提问轨道
                  </div>
                  <div className="prompt-row prompt-rail-grid">
                    {starterPrompts.map((prompt) => (
                      <button
                        key={prompt}
                        type="button"
                        className="prompt-chip"
                        onClick={() => usePrompt(prompt)}
                        disabled={loading}
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>

                {error ? (
                  <div className="error-box hero-status-box">
                    <AlertTriangle size={16} />
                    {error}
                  </div>
                ) : (
                  <div className={`status-banner hero-status-box ${loading ? 'pending' : 'idle'}`}>
                    {loading ? <Loader2 size={16} className="spinning" /> : <Clock3 size={16} />}
                    {loading
                      ? 'Agent 正在读取市场、新闻、历史类比并生成三周期判断…'
                      : '输入越具体，系统越容易把判断收敛成你真正能执行的下一步。'}
                  </div>
                )}
              </motion.form>

              <motion.button variants={itemVariants} type="button" className="hero-scroll-cue" onClick={scrollToBriefing}>
                <span>向下查看完整 briefing 与证据</span>
                <ChevronDown size={16} />
              </motion.button>
            </motion.div>

            <motion.aside className="hero-intelligence" initial="hidden" animate="show" variants={containerVariants}>
              <motion.article variants={itemVariants} className="intel-card intel-card-primary">
                <div className="intel-card-head">
                  <span className="intel-label">Current Thesis</span>
                  <span className={`intel-tone ${summary ? stanceTone[summary.stance] : 'tone-outline'}`}>
                    {summary ? summary.confidence_band : '待生成'}
                  </span>
                </div>
                <strong>{summary ? summary.stance : '等待分析'}</strong>
                <p>{summary ? summary.reasons[0] : '提交一个更具体的问题后，这里会先亮出本轮主结论与建议动作。'}</p>
                <div className="intel-chip-row">
                  <span className={`signal-chip ${summary ? stanceTone[summary.stance] : 'tone-outline'}`}>
                    {summary?.action || '准备中'}
                  </span>
                  <span className="signal-chip tone-outline">
                    {summary ? horizonLabels[summary.horizon] : horizonLabels[horizon]}
                  </span>
                </div>
              </motion.article>

              <motion.article variants={itemVariants} className="intel-card">
                <div className="intel-card-head">
                  <span className="intel-label">Signal Grid</span>
                  <TrendingUp size={16} />
                </div>
                <div className="hero-stat-grid">
                  {analysisStats.map((item) => (
                    <InsightStat key={item.label} label={item.label} value={item.value} tone="hero" />
                  ))}
                </div>
              </motion.article>

              <motion.article variants={itemVariants} className="intel-card">
                <div className="intel-card-head">
                  <span className="intel-label">Workflow</span>
                  <Waves size={16} />
                </div>
                <ol className="hero-steps">
                  {workflowSteps.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              </motion.article>

              <motion.article variants={itemVariants} className="intel-card intel-card-preview">
                <div className="intel-card-head">
                  <span className="intel-label">Question Preview</span>
                  <Newspaper size={16} />
                </div>
                <p className="intel-preview-text">{deferredQuestion || '先输入你的问题，再让 Agent 开始分析。'}</p>
                <div className="intel-split">
                  <div>
                    <span className="intel-label">锁定画像</span>
                    <strong>{riskLabels[riskProfile]}</strong>
                  </div>
                  <div>
                    <span className="intel-label">分析镜头</span>
                    <strong>{horizonTags[horizon]}</strong>
                  </div>
                </div>
              </motion.article>
            </motion.aside>
          </div>
        </div>
      </section>

      <section className="content-shell">
        <div className="content-inner">
          <div className="section-ribbon">
            <span className="section-ribbon-label">Today&apos;s Lens</span>
            <div className="section-ribbon-chips">
              <span className={`signal-chip ${summary ? stanceTone[summary.stance] : 'tone-outline'}`}>{quickMood}</span>
              <span className="signal-chip tone-outline">主观察周期：{horizonLabels[horizon]}</span>
              <span className="signal-chip tone-outline">当前画像：{riskLabels[riskProfile]}</span>
            </div>
          </div>

          <section className="workspace-grid">
            <motion.section
              ref={briefingRef}
              className="panel panel-briefing"
              initial="hidden"
              animate="show"
              variants={containerVariants}
            >
              <motion.div variants={itemVariants} className="panel-head">
                <div>
                  <h2>本轮判断 Briefing</h2>
                  <p>先把结论、风险和理由讲清楚，再决定是否继续深挖。</p>
                </div>
                <div className={`timing-chip ${loading ? 'pending' : ''}`}>
                  {loading ? <Loader2 size={14} className="spinning" /> : <Clock3 size={14} />}
                  {analysis?.timing_ms?.total ? `${analysis.timing_ms.total} ms` : '未执行'}
                </div>
              </motion.div>

              {loading && !summary ? (
                <motion.div variants={itemVariants} className="loading-panel">
                  <div className="decision-banner waiting">
                    <div className="decision-copy">
                      <span className="summary-kicker">处理中</span>
                      <h3>Agent 正在生成本轮结论</h3>
                      <p>这一步会把市场快照、量化预测、新闻和历史类比压缩成一张可读的结论卡。</p>
                    </div>
                    <Loader2 size={28} className="spinning" />
                  </div>
                  <div className="loading-grid">
                    {['读取快照', '抓取新闻', '合成判断', '收敛建议'].map((item) => (
                      <div key={item} className="loading-card">
                        <span>{item}</span>
                        <div className="loading-line" />
                        <div className="loading-line short" />
                      </div>
                    ))}
                  </div>
                </motion.div>
              ) : summary ? (
                <>
                  <motion.div variants={itemVariants} className="decision-banner">
                    <div className="decision-copy">
                      <span className="summary-kicker">主结论</span>
                      <h3>{summary.action}</h3>
                      <p>{summary.reasons[0]}</p>
                    </div>
                    <div className={`decision-tone ${stanceTone[summary.stance] || 'tone-neutral'}`}>
                      <strong>{summary.stance}</strong>
                      <span>{summary.confidence_band}置信度</span>
                    </div>
                  </motion.div>

                  <motion.div variants={itemVariants} className="briefing-stats">
                    {analysisStats.map((item) => (
                      <InsightStat key={item.label} label={item.label} value={item.value} />
                    ))}
                  </motion.div>

                  <motion.div variants={itemVariants} className="summary-main">
                    <div>
                      <span className="summary-kicker">当前倾向</span>
                      <strong>{summary.stance}</strong>
                    </div>
                    <div>
                      <span className="summary-kicker">建议动作</span>
                      <strong>{summary.action}</strong>
                    </div>
                    <div>
                      <span className="summary-kicker">观察周期</span>
                      <strong>{horizonLabels[summary.horizon]}</strong>
                    </div>
                  </motion.div>

                  <motion.div variants={itemVariants} className="summary-columns">
                    <SummaryList title="为什么这么看" items={summary.reasons} />
                    <SummaryList title="什么情况下失效" items={summary.invalidators} />
                  </motion.div>

                  <motion.div variants={itemVariants} className="briefing-sidecar">
                    <div className={`risk-banner ${riskBanner?.level || 'medium'}`}>
                      <strong>{riskBanner?.title || '等待分析'}</strong>
                      <p>{riskBanner?.message || '执行分析后，这里会显示本轮最关键的风险提醒。'}</p>
                    </div>

                    <div className="focus-card">
                      <div className="focus-card-head">
                        <span>你当前关注的周期</span>
                        <strong>{selectedForecast ? horizonLabels[selectedForecast.horizon] : horizonLabels[horizon]}</strong>
                      </div>
                      {selectedForecast ? (
                        <>
                          <div className={`signal-chip ${stanceTone[selectedForecast.stance] || 'tone-neutral'}`}>
                            {selectedForecast.stance}
                          </div>
                          <p>
                            {selectedForecast.action}，{selectedForecast.confidence_band}置信度，概率约{' '}
                            {(selectedForecast.probability * 100).toFixed(1)}%。
                          </p>
                          <span className="focus-note">{basisLabels[selectedForecast.basis] || selectedForecast.basis}</span>
                        </>
                      ) : (
                        <p>分析完成后，这里会突出当前所选周期的重点判断。</p>
                      )}
                    </div>
                  </motion.div>

                  <motion.p variants={itemVariants} className="disclaimer">
                    {summary.disclaimer}
                  </motion.p>

                  <motion.div variants={itemVariants} className="feedback-row">
                    <span>这条回答对你有帮助吗？</span>
                    <button
                      type="button"
                      className="feedback-button"
                      onClick={() => submitFeedback('helpful')}
                      disabled={feedbackLoading}
                    >
                      有帮助
                    </button>
                    <button
                      type="button"
                      className="feedback-button ghost"
                      onClick={() => submitFeedback('not_helpful')}
                      disabled={feedbackLoading}
                    >
                      没帮助
                    </button>
                  </motion.div>
                  {feedbackStatus ? (
                    <motion.p variants={itemVariants} className="feedback-status">
                      {feedbackStatus}
                    </motion.p>
                  ) : null}

                  <motion.div variants={itemVariants} className="follow-up-block">
                    <div className="section-mini-head">
                      <div>
                        <h3>继续追问</h3>
                        <span>点一下就能把问题带回输入区，继续推进这一轮分析。</span>
                      </div>
                    </div>
                    <div className="follow-up-grid">
                      {followUpQuestions.map((item) => (
                        <button key={item} type="button" className="follow-up-button" onClick={() => handleFollowUp(item)}>
                          <span>{item}</span>
                          <ArrowRight size={16} />
                        </button>
                      ))}
                    </div>
                  </motion.div>
                </>
              ) : (
                <motion.div variants={itemVariants} className="empty-briefing">
                  <BrainCircuit size={22} />
                  <div>
                    <h3>先发起一次问题</h3>
                    <p>系统会返回“方向 + 风险 + 证据 + 失效条件”的完整回答，并把三周期判断一起展开。</p>
                  </div>
                </motion.div>
              )}
            </motion.section>

            <motion.section className="panel panel-sidecar" initial="hidden" animate="show" variants={containerVariants}>
              <motion.div variants={itemVariants} className="panel-head">
                <div>
                  <h2>上下文与自动情报</h2>
                  <p>系统自动抓取的新闻、你的最近问题与当前这轮交互节奏。</p>
                </div>
                <MessagesSquare size={18} />
              </motion.div>

              <div className="side-stack">
                <motion.div variants={itemVariants} className="side-surface auto-news-block">
                  <div className="section-mini-head">
                    <div>
                      <h3>系统自动抓取的新闻</h3>
                      <span>{recentNews.length ? `${recentNews.length} 条已纳入本轮` : '提交问题后自动抓取'}</span>
                    </div>
                    <div className="mini-badge">无需手动贴新闻</div>
                  </div>
                  <div className="news-brief-list">
                    {recentNews.length === 0 ? (
                      <div className="placeholder-card">
                        <BrainCircuit size={18} />
                        提交问题后，系统会自动获取最近的黄金/宏观新闻，并把它们融入分析。
                      </div>
                    ) : (
                      recentNews.slice(0, 3).map((item) => (
                        <article key={item.event_id} className="news-brief-card">
                          <div className="news-brief-meta">
                            <span>{item.source}</span>
                            <span>{formatDateTime(item.published_at)}</span>
                          </div>
                          <h3>{item.title}</h3>
                          <p>{item.summary}</p>
                        </article>
                      ))
                    )}
                  </div>
                </motion.div>

                <motion.div variants={itemVariants} className="side-surface conversation-block">
                  <div className="section-mini-head">
                    <div>
                      <h3>最近对话</h3>
                      <span>保留最近几轮提问，方便你继续追问与修正视角。</span>
                    </div>
                  </div>

                  <div className="chat-stack">
                    {conversation.length === 0 ? (
                      <div className="chat-empty">
                        <BrainCircuit size={20} />
                        <p>提交一次问题后，这里会留下你的提问和本轮最核心的一句判断。</p>
                      </div>
                    ) : (
                      conversation.map((item) => (
                        <div key={item.id} className={`chat-bubble ${item.role}`}>
                          <span>{item.role === 'user' ? '你' : 'GoldenSense'}</span>
                          <p>{item.content}</p>
                        </div>
                      ))
                    )}
                  </div>
                </motion.div>
              </div>
            </motion.section>
          </section>

          <section className="insight-grid">
            <motion.section className="panel" initial="hidden" animate="show" variants={containerVariants}>
              <motion.div variants={itemVariants} className="panel-head">
                <div>
                  <h2>三周期参考</h2>
                  <p>同一批新闻与市场快照下的 T+1 / T+7 / T+30，对比后更容易看清短中期分歧。</p>
                </div>
                <BadgeAlert size={18} />
              </motion.div>

              <motion.div variants={itemVariants} className="multi-horizon-grid">
                {horizonForecasts.length === 0 ? (
                  <div className="placeholder-card">
                    <Clock3 size={18} />
                    分析完成后，这里会同时给出 T+1、T+7、T+30 的判断与原因。
                  </div>
                ) : (
                  horizonForecasts.map((item) => (
                    <article
                      key={item.horizon}
                      className={`horizon-card ${item.horizon === horizon ? 'active' : ''} ${stanceTone[item.stance] || 'tone-neutral'}`}
                    >
                      <div className="horizon-card-head">
                        <span>{horizonTags[item.horizon]}</span>
                        <span>{basisLabels[item.basis] || item.basis}</span>
                      </div>
                      <strong>{item.stance}</strong>
                      <div className="horizon-card-stats">
                        <span>{item.action}</span>
                        <span>{item.confidence_band}置信度</span>
                        <span>{(item.probability * 100).toFixed(1)}%</span>
                      </div>
                      <ul>
                        {item.reasons.slice(0, 3).map((reason) => (
                          <li key={reason}>{reason}</li>
                        ))}
                      </ul>
                    </article>
                  ))
                )}
              </motion.div>
            </motion.section>

            <motion.section className="panel" initial="hidden" animate="show" variants={containerVariants}>
              <motion.div variants={itemVariants} className="panel-head">
                <div>
                  <h2>证据卡片</h2>
                  <p>每张卡只讲一个观点，并且可以直接跳到对应引用，不让“结论”脱离证据。</p>
                </div>
                <FileSearch size={18} />
              </motion.div>

              <motion.div variants={itemVariants} className="evidence-grid">
                {evidenceCards.length === 0 ? (
                  <div className="placeholder-card">
                    <FileSearch size={18} />
                    分析完成后，这里会把市场、量化、新闻、历史和风险拆成独立证据卡。
                  </div>
                ) : (
                  evidenceCards.map((card) => {
                    const Icon = signalTypeMeta[card.signal_type]?.icon || FileSearch;
                    return (
                      <div key={card.id} className={`evidence-card ${card.direction}`}>
                        <div className="evidence-head">
                          <span className="evidence-label">
                            <Icon size={14} />
                            {signalTypeMeta[card.signal_type]?.label || card.signal_type}
                          </span>
                          <span className="evidence-direction">{card.direction}</span>
                        </div>
                        <h3>{card.title}</h3>
                        <p>{card.takeaway}</p>
                        <div className="citation-pills">
                          {card.citation_ids.map((id) => (
                            <button key={id} type="button" onClick={() => scrollToCitation(id)}>
                              #{id}
                            </button>
                          ))}
                        </div>
                      </div>
                    );
                  })
                )}
              </motion.div>
            </motion.section>
          </section>

          <section className="panel citation-panel">
            <div className="panel-head">
              <div>
                <h2>可追溯引用</h2>
                <p>所有用户可见结论都能回到证据文本，而不是只留下一个模糊结论。</p>
              </div>
              <ShieldCheck size={18} />
            </div>

            <div className="citation-list">
              {citations.length === 0 ? (
                <div className="placeholder-card">
                  <ShieldCheck size={18} />
                  当分析结果返回后，引用会集中展示在这里。
                </div>
              ) : (
                citations.map((citation) => (
                  <article
                    key={citation.id}
                    id={`citation-${citation.id}`}
                    className={`citation-item ${highlightedCitation === citation.id ? 'highlighted' : ''}`}
                  >
                    <div className="citation-meta">
                      <span>{citation.id}</span>
                      <span>{citation.source_type}</span>
                    </div>
                    <h3>{citation.label}</h3>
                    <p>{citation.excerpt}</p>
                    {citation.url ? (
                      <a href={citation.url} target="_blank" rel="noreferrer">
                        打开原始来源
                        <ExternalLink size={14} />
                      </a>
                    ) : null}
                  </article>
                ))
              )}
            </div>
          </section>
        </div>
      </section>
    </div>
  );
}

function SelectionBadge({ label, value }) {
  return (
    <div className="selection-badge">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SegmentedControl({ label, value, options, description, disabled, onChange }) {
  return (
    <div className="control-cluster">
      <div className="control-head">
        <span>{label}</span>
        <strong>{options.find((item) => item.value === value)?.label}</strong>
      </div>
      <div className="segmented-control">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            className={value === option.value ? 'segmented-option active' : 'segmented-option'}
            onClick={() => onChange(option.value)}
            disabled={disabled}
          >
            {option.label}
          </button>
        ))}
      </div>
      <motion.p
        key={value}
        className="control-description"
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
      >
        {description}
      </motion.p>
    </div>
  );
}

function InsightStat({ label, value, tone = 'light' }) {
  return (
    <div className={`insight-stat ${tone}`}>
      <span>{label}</span>
      <strong>
        <AnimatedValue value={value} />
      </strong>
    </div>
  );
}

function AnimatedValue({ value }) {
  const shouldReduceMotion = useReducedMotion();
  const parsed = useMemo(() => parseNumericDisplay(value), [value]);
  const [displayValue, setDisplayValue] = useState(value);

  useEffect(() => {
    if (!parsed || shouldReduceMotion) {
      setDisplayValue(value);
      return undefined;
    }

    let frameId = 0;
    let startTime = 0;
    const durationMs = 720;

    const animate = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / durationMs, 1);
      const eased = 1 - (1 - progress) * (1 - progress);
      const currentValue = parsed.target * eased;
      setDisplayValue(formatNumericDisplay(parsed, currentValue));

      if (progress < 1) {
        frameId = window.requestAnimationFrame(animate);
      }
    };

    frameId = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(frameId);
  }, [parsed, shouldReduceMotion, value]);

  return displayValue;
}

function SummaryList({ title, items }) {
  return (
    <div className="summary-list">
      <h4>{title}</h4>
      <ul>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

function parseNumericDisplay(value) {
  const match = /(-?\d+(?:\.\d+)?)/.exec(value);
  if (!match) return null;

  const numericPortion = match[1];
  return {
    prefix: value.slice(0, match.index),
    suffix: value.slice(match.index + numericPortion.length),
    target: Number.parseFloat(numericPortion),
    decimals: numericPortion.includes('.') ? numericPortion.split('.')[1].length : 0,
  };
}

function formatNumericDisplay(parsed, currentValue) {
  const safeValue = currentValue > parsed.target ? parsed.target : currentValue;
  return `${parsed.prefix}${safeValue.toFixed(parsed.decimals)}${parsed.suffix}`;
}

function formatDateTime(value) {
  try {
    return new Date(value).toLocaleString('zh-CN', { hour12: false });
  } catch (error) {
    return value || '未知时间';
  }
}

export default App;
