import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  BookOpenCheck,
  CalendarClock,
  CheckCircle2,
  FileClock,
  Fingerprint,
  Loader2,
  Mail,
  Radar,
  ScrollText,
  ShieldCheck,
  Target,
} from 'lucide-react';
import {
  deriveApiEndpoint,
  formatPublicationDate,
  isSignalPublicationEmpty,
  isSignalPublication,
} from './researchContract';

import { horizonLabel, RESEARCH_HORIZON_WINDOWS } from './horizons';
import { ActiveCaseRibbon, CaseStrategyPanel } from './ResearchCasePanel';
import { useViewMode } from './viewMode';

const API_URL = import.meta.env.VITE_AGENT_API_URL || '/api/v1/agent/analyze';
const MARKET_VIEW_URL =
  import.meta.env.VITE_AGENT_MARKET_VIEW_URL || API_URL.replace('/analyze', '/market-view');
const SIGNALS_BASE =
  import.meta.env.VITE_SIGNALS_BASE_URL || deriveApiEndpoint(API_URL, '/api/v1/signals');
const SUBSCRIBE_URL =
  import.meta.env.VITE_SUBSCRIBE_URL || deriveApiEndpoint(API_URL, '/api/v1/subscriptions');
const RESEARCH_URL =
  import.meta.env.VITE_AGENT_RESEARCH_URL || API_URL.replace('/analyze', '/research/current');
const API_KEY = import.meta.env.VITE_AGENT_API_KEY || 'dev-public-key';

const horizonIcons = {
  short_term: Radar,
  mid_term: Target,
  long_term: BookOpenCheck,
};

const confidenceTone = { 高: 'bull', 中: 'neutral', 低: 'risk' };
const CURRENT_PUBLICATION_UNAVAILABLE = {
  status: 'unavailable',
  guidance: '当前无法确认本周发布状态；不会展示未经验证的发布记录。',
};
const HISTORY_UNAVAILABLE = { status: 'unavailable', publications: [] };
const TRACK_RECORD_UNAVAILABLE = { status: 'unavailable', record: null };

function headers() {
  return { 'X-API-Key': API_KEY };
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: headers() });
  const text = await response.text();
  let json = null;
  if (text.trim()) {
    try {
      json = JSON.parse(text);
    } catch {
      throw new Error('研究服务暂时返回了无法识别的内容，请稍后重试。');
    }
  }
  if (!response.ok) {
    const detail = json?.detail;
    throw new Error(
      (typeof detail === 'string' ? detail : detail?.message) || `HTTP ${response.status}`,
    );
  }
  return json;
}

export default function SignalsPage() {
  const { mode } = useViewMode();
  const isPro = mode === 'pro';
  const [book, setBook] = useState(null);
  const [current, setCurrent] = useState(null);
  const [historyState, setHistoryState] = useState({ status: 'loading', publications: [] });
  const [trackRecordState, setTrackRecordState] = useState({ status: 'loading', record: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        setError('');
        const [bookResult, currentResult, historyResult, trackRecordResult] = await Promise.allSettled([
          fetchJson(MARKET_VIEW_URL),
          fetchJson(`${SIGNALS_BASE}/current`)
            .then((payload) => (
              isSignalPublication(payload) || isSignalPublicationEmpty(payload)
                ? payload
                : CURRENT_PUBLICATION_UNAVAILABLE
            )),
          fetchJson(`${SIGNALS_BASE}/history?limit=26`)
            .then((payload) => (
              Array.isArray(payload?.publications)
                ? { status: 'available', publications: payload.publications }
                : HISTORY_UNAVAILABLE
            )),
          fetchJson(`${SIGNALS_BASE}/track-record`)
            .then((payload) => (
              payload
              && typeof payload === 'object'
              && !Array.isArray(payload)
              && payload.per_profile
              && payload.benchmarks
                ? { status: 'available', record: payload }
                : TRACK_RECORD_UNAVAILABLE
            )),
        ]);
        if (cancelled) return;
        if (bookResult.status === 'fulfilled') {
          setBook(bookResult.value);
        } else {
          setBook(null);
          setError(bookResult.reason?.message || '观点书读取失败');
        }
        setCurrent(
          currentResult.status === 'fulfilled'
            ? currentResult.value
            : CURRENT_PUBLICATION_UNAVAILABLE,
        );
        setHistoryState(
          historyResult.status === 'fulfilled' ? historyResult.value : HISTORY_UNAVAILABLE,
        );
        setTrackRecordState(
          trackRecordResult.status === 'fulfilled'
            ? trackRecordResult.value
            : TRACK_RECORD_UNAVAILABLE,
        );
      } catch (loadError) {
        if (!cancelled) setError(loadError.message || '观点书读取失败');
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const meta = book?.meta;
  const history = historyState.publications;
  const historyUnavailable = historyState.status === 'unavailable';
  const trackRecord = trackRecordState.record;
  const trackRecordUnavailable = trackRecordState.status === 'unavailable';
  const hasCurrentPublication = isSignalPublication(current);
  const currentPublicationUnavailable = current?.status === 'unavailable';
  const publicationDate = formatPublicationDate(current?.published_at);
  const ledgerIsEmpty = isSignalPublicationEmpty(current);

  return (
    <main className="page-surface signals-page">
      <section className="terminal-header">
        <div>
          <p className="eyebrow">Market View Book & Signal Ledger</p>
          <h1>大盘观点书与每周信号台账</h1>
          <p>
            短/中/长三时间尺度的系统观点（每节自带失效条件），以及每周冻结、不可重写的
            发布记录与前向影子组合记分卡——回测归回测，前向归前向。
          </p>
        </div>
        <div className="compliance-pill">
          <ShieldCheck size={15} />
          append-only · 不回填
        </div>
      </section>

      <ActiveCaseRibbon />

      <CaseStrategyPanel />

      <section className="truth-class-grid" aria-label="结果证据类型">
        <article>
          <span className="truth-class-badge class-backtest">历史回测</span>
          <strong>模型实验结果</strong>
          <small>只说明历史样本表现，不冒充未来兑现。</small>
        </article>
        <article>
          <span className="truth-class-badge class-simulated">模拟前向</span>
          <strong>影子组合记分</strong>
          <small>发布后按假设组合计分，不是真实账户收益。</small>
        </article>
        <article>
          <span className="truth-class-badge class-live">真实前向</span>
          <strong>不可变周度发布</strong>
          <small>
            {hasCurrentPublication
              ? '当前 ' + current.publication_id
              : ledgerIsEmpty
                ? '等待第一期发布，不回填历史。'
                : '本周发布状态暂不可确认。'}
          </small>
        </article>
      </section>

      {meta?.data_asof ? (
        <div className={`freshness-banner ${meta.data_stale ? 'stale' : ''}`}>
          <CalendarClock size={14} />
          数据截至 {meta.data_asof}{meta.data_age_days != null ? ` · ${meta.data_age_days} 天前收盘` : ''} · 非实时行情
          {meta.data_stale ? ' · 已超新鲜度阈值' : ''}
        </div>
      ) : null}

      {error ? (
        <div className="error-panel">
          <AlertTriangle size={18} />
          <div>
            <strong>观点书不可用</strong>
            <p>{error}</p>
          </div>
        </div>
      ) : null}

      {loading ? (
        <div className="placeholder-panel">
          <Loader2 size={20} className="spinning" />
          <span>正在组装三尺度观点书与台账记录。</span>
        </div>
      ) : null}

      {book ? (
        <section className="view-book-grid">
          {Object.entries(horizonIcons).map(([key, Icon]) => {
            const section = book[key];
            return (
              <article key={key} className={`view-book-card ${section?.available ? '' : 'degraded'}`}>
                <div className="view-book-head">
                  <span>
                    <Icon size={15} /> {horizonLabel(key)} · {RESEARCH_HORIZON_WINDOWS[key]}
                  </span>
                  {section?.available ? (
                    <span className={`stance-badge tone-${confidenceTone[section.confidence] || 'neutral'}`}>
                      置信度 {section.confidence}
                    </span>
                  ) : (
                    <span className="stance-badge tone-risk">降级</span>
                  )}
                </div>
                {section?.available ? (
                  <>
                    <p className="view-book-core">{section.core_view}</p>
                    {isPro ? (
                      <div className="view-book-evidence">
                        {(section.evidence || []).slice(0, 3).map((line) => (
                          <span key={line}>{line}</span>
                        ))}
                      </div>
                    ) : null}
                    <div className="view-book-invalidation">
                      <strong>失效条件</strong>
                      <ul>
                        {(section.invalidation || []).map((line) => (
                          <li key={line}>{line}</li>
                        ))}
                      </ul>
                    </div>
                  </>
                ) : (
                  <p className="muted-copy">该节数据降级：{section?.degraded_reason}</p>
                )}
              </article>
            );
          })}
        </section>
      ) : null}

      <section className="ledger-layout">
        <section className="panel-block ledger-current">
          <div className="panel-title">
            <ScrollText size={16} />
            <div>
              <h2>本周发布</h2>
              <span>
                {hasCurrentPublication
                  ? current.publication_id
                  : ledgerIsEmpty
                    ? '台账为空'
                    : loading ? '正在读取' : '状态暂不可用'}
              </span>
            </div>
          </div>
          {hasCurrentPublication ? (
            <>
              <div className="ledger-alloc-grid">
                {Object.entries(current.allocations || {}).map(([profile, alloc]) => (
                  <div key={profile} className="ledger-alloc-card">
                    <span>{{ conservative: '保守', balanced: '稳健', aggressive: '进取' }[profile]}</span>
                    <strong>
                      {alloc.range_pct ? `${alloc.range_pct[0]}%–${alloc.range_pct[1]}%` : 'N/A'}
                    </strong>
                    <small>中点 {alloc.midpoint ?? 'N/A'}%</small>
                  </div>
                ))}
              </div>
              {isPro ? (
                <div className="ledger-hash">
                  <Fingerprint size={13} />
                  <code>{current.content_hash?.slice(0, 26)}…</code>
                  {publicationDate ? <span>发布于 {publicationDate}</span> : null}
                </div>
              ) : null}
              <p className="disclaimer">{current.disclaimer}</p>
            </>
          ) : ledgerIsEmpty || currentPublicationUnavailable ? (
            <div className="ledger-empty-guidance">
              <strong>
                {currentPublicationUnavailable ? '本周发布暂不可用' : '需要第一期真实前向发布'}
              </strong>
              <p>{current?.guidance || '追踪记录自首次发布起前向累积，不回填历史。'}</p>
              <small>
                {currentPublicationUnavailable
                  ? '请稍后重试；异常响应不会被视为真实发布。'
                  : '运维动作：冻结本周研究后执行内部发布任务。'}
              </small>
            </div>
          ) : null}
        </section>

        <section className="panel-block ledger-track">
          <div className="panel-title">
            <FileClock size={16} />
            <div>
              <h2>前向记分卡</h2>
              <span>
                {trackRecordUnavailable
                  ? '记分状态暂不可用'
                  : trackRecord?.matured_through
                  ? `已成熟至 ${trackRecord.matured_through} · 成本 ${trackRecord.cost_bps}bps`
                  : trackRecord
                    ? '暂无成熟周（诚实空账）'
                    : '读取中'}
              </span>
            </div>
          </div>
          {trackRecordUnavailable ? (
            <p className="muted-copy">记分服务暂不可用。</p>
          ) : trackRecord ? (
            <div className="track-grid">
              {trackRecord.price_evidence === 'session_timestamps' && (
                <p className="muted-copy">行情按交易会话和报价可用时间评分。</p>
              )}
              {trackRecord.price_evidence === 'date_only_proxy' && (
                <p className="muted-copy">行情按日期估算可用时间，记分供研究参考。</p>
              )}
              {Object.entries(trackRecord.per_profile || {}).map(([profile, stats]) => (
                <div key={profile} className="track-row">
                  <strong>{{ conservative: '保守', balanced: '稳健', aggressive: '进取' }[profile]}</strong>
                  <span>周数 {stats.weeks_scored}</span>
                  <span>
                    累计 {stats.cum_return != null ? `${(stats.cum_return * 100).toFixed(2)}%` : '—'}
                  </span>
                  <span>
                    回撤 {stats.max_drawdown != null ? `${(stats.max_drawdown * 100).toFixed(2)}%` : '—'}
                  </span>
                  {trackRecord.benchmarks?.[`${profile}_fixed_net`]?.cum_return != null && (
                    <span className="track-row-benchmark">同成本固定仓位 {(trackRecord.benchmarks[`${profile}_fixed_net`].cum_return * 100).toFixed(2)}%</span>
                  )}
                </div>
              ))}
              <div className="track-benchmarks">
                <span>
                  基准·金价买入持有：
                  {trackRecord.benchmarks?.gold_buy_hold?.cum_return != null
                    ? `${(trackRecord.benchmarks.gold_buy_hold.cum_return * 100).toFixed(2)}%`
                    : '—'}
                </span>
              </div>
              <p className="disclaimer">证据类型：模拟前向 · {trackRecord.disclaimer}</p>
            </div>
          ) : null}
        </section>
      </section>

      {!isPro ? (
        <p className="muted-copy simple-mode-hint">
          简明模式已折叠证据链、防篡改哈希与发布历史审计表——右上角切换「专业」查看完整审计链。
        </p>
      ) : null}

      {isPro ? (
      <section className="panel-block ledger-history">
        <div className="panel-title">
          <ScrollText size={16} />
          <div>
            <h2>发布历史（审计链）</h2>
            <span>{historyUnavailable ? '状态暂不可用' : `${history.length} 期 · append-only`}</span>
          </div>
        </div>
        {historyUnavailable ? (
          <p className="muted-copy">发布历史暂不可用，请稍后重试。</p>
        ) : history.length ? (
          <div className="history-table" role="table">
            <div className="history-row history-head" role="row">
              <span>期号</span>
              <span>数据截至</span>
              <span>稳健区间</span>
              <span>哈希</span>
            </div>
            {[...history].reverse().map((record) => (
              <div key={record.publication_id} className="history-row" role="row">
                <span>{record.publication_id}</span>
                <span>{record.data_asof}</span>
                <span>
                  {record.allocations?.balanced?.range_pct
                    ? `${record.allocations.balanced.range_pct[0]}%–${record.allocations.balanced.range_pct[1]}%`
                    : 'N/A'}
                </span>
                <code>{record.content_hash?.slice(7, 19)}</code>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted-copy">尚无发布记录。</p>
        )}
      </section>
      ) : null}

      <CrossAssetPanel />

      <DigestSubscribeCard />

      <footer className="terminal-footer">
        <span>GoldenSense Signal Ledger</span>
        <span>不可变发布 · 成熟周计分 · 双基准对照 · 非投资建议</span>
      </footer>
    </main>
  );
}

function CrossAssetPanel() {
  const [cross, setCross] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetch(RESEARCH_URL, { headers: headers(), signal: controller.signal })
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then((payload) => setCross(payload.cross_asset || null))
      .catch(() => {
        if (!controller.signal.aborted) setFailed(true);
      });
    return () => controller.abort();
  }, []);

  return (
    <section className="panel-block cross-asset-panel">
      <div className="panel-title">
        <Radar size={16} />
        <div>
          <h2>跨资产背景</h2>
          <span>
            {cross
              ? `${cross.corr_window_days} 日滚动相关 · 近一年表现对照（历史统计，非配置信号）`
              : failed
                ? '跨资产数据暂不可用'
                : '读取中'}
          </span>
        </div>
      </div>
      {cross ? (
        <>
          <div className="cross-asset-grid">
            <div className="cross-asset-row cross-asset-head">
              <span>资产</span>
              <span>与黄金相关性</span>
              <span>近一年表现</span>
            </div>
            <div className="cross-asset-row">
              <span>黄金（基准）</span>
              <span>—</span>
              <strong>{(cross.gold_perf_1y * 100).toFixed(1)}%</strong>
            </div>
            {Object.entries(cross.peers).map(([key, peer]) => (
              <div key={key} className="cross-asset-row">
                <span>{peer.label}</span>
                <span className="corr-cell">
                  <i
                    className={peer.corr_63d >= 0 ? 'corr-pos' : 'corr-neg'}
                    style={{ width: `${Math.min(100, Math.abs(peer.corr_63d || 0) * 100)}%` }}
                  />
                  <small>{peer.corr_63d != null ? peer.corr_63d.toFixed(2) : 'N/A'}</small>
                </span>
                <strong>{(peer.perf_1y * 100).toFixed(1)}%</strong>
              </div>
            ))}
          </div>
          <p className="muted-copy">{cross.note}</p>
        </>
      ) : null}
    </section>
  );
}

function DigestSubscribeCard() {
  const [email, setEmail] = useState('');
  const [state, setState] = useState('idle'); // idle | sending | done | error
  const [message, setMessage] = useState('');

  async function submit(event) {
    event.preventDefault();
    setState('sending');
    setMessage('');
    try {
      const response = await fetch(SUBSCRIBE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
        body: JSON.stringify({ email: email.trim() }),
      });
      const text = await response.text();
      let json = null;
      try {
        json = text ? JSON.parse(text) : null;
      } catch {
        throw new Error('订阅服务暂时返回异常，请稍后重试。');
      }
      if (!response.ok) {
        throw new Error(json?.detail?.message || `订阅失败：HTTP ${response.status}`);
      }
      setState('done');
      setMessage(
        json.created
          ? `已订阅（${json.email_masked}）：每周发布日自动送达，邮件内含一键退订链接。`
          : `该邮箱已在订阅列表中（${json.email_masked}）。`,
      );
    } catch (submitError) {
      setState('error');
      setMessage(submitError.message || '订阅失败');
    }
  }

  return (
    <section className="panel-block digest-subscribe">
      <div className="panel-title">
        <Mail size={16} />
        <div>
          <h2>订阅每周信号</h2>
          <span>发布日自动送达 · 邮箱仅存本地 · 一键退订</span>
        </div>
      </div>
      {state === 'done' ? (
        <p className="subscribe-ok">
          <CheckCircle2 size={14} /> {message}
        </p>
      ) : (
        <form className="subscribe-row" onSubmit={submit}>
          <input
            type="email"
            required
            placeholder="you@example.com"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            aria-label="订阅邮箱"
          />
          <button type="submit" disabled={state === 'sending'}>
            {state === 'sending' ? <Loader2 size={14} className="spinning" /> : '订阅'}
          </button>
        </form>
      )}
      {state === 'error' ? <p className="subscribe-err">{message}</p> : null}
    </section>
  );
}
