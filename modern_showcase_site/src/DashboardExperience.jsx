import React from 'react';
import { AlertTriangle, Clock3, ExternalLink, Newspaper, RefreshCw, TrendingDown, TrendingUp } from 'lucide-react';

function formatPrice(value) {
  return Number.isFinite(Number(value))
    ? `$${Number(value).toLocaleString('en-US', { maximumFractionDigits: 2 })}`
    : '暂无';
}

function formatPercent(value) {
  if (!Number.isFinite(Number(value))) return '暂无';
  const percent = Number(value) * 100;
  return `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;
}

function formatTime(value) {
  if (!value) return '时间未知';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('zh-CN', { hour12: false });
}

export function DataStateBoundary({ state, onRetry, children }) {
  if (state.kind === 'unavailable') {
    return (
      <div className="dashboard-state-error" role="alert">
        <AlertTriangle size={20} />
        <div>
          <strong>{state.message?.summary || '暂时无法获取当前市场数据。'}</strong>
          <p>已停止展示由当前行情推导的结论，避免把旧数据当作实时信息。</p>
        </div>
        <button type="button" onClick={onRetry}>
          <RefreshCw size={15} />
          重新加载
        </button>
      </div>
    );
  }
  return children;
}

export function MarketSnapshotHero({ dashboard, state, onRetry }) {
  const market = dashboard?.market_status;
  const forecast = dashboard?.horizon_forecasts?.[0];
  const DirectionIcon = Number(market?.price_change_pct_1d) >= 0 ? TrendingUp : TrendingDown;
  return (
    <section className={`market-snapshot-hero state-${state.kind}`} data-dashboard-section="market-snapshot" aria-labelledby="market-snapshot-title" aria-label="Market summary">
      <div className="dashboard-section-head">
        <div>
          <span className="section-index">01 · LIVE CONTEXT</span>
          <h2 id="market-snapshot-title">市场快照</h2>
          <p>先看当前价格、时点和数据状态，再看研究判断。</p>
        </div>
        <span className={`dashboard-state-pill tone-${state.tone}`}>{state.label}</span>
      </div>
      <DataStateBoundary state={state} onRetry={onRetry}>
        <div className="snapshot-grid" aria-label="Market summary">
          <article className="snapshot-primary">
            <span>XAUUSD</span>
            <strong>{state.kind === 'loading' ? '读取中' : formatPrice(market?.latest_price)}</strong>
            <small><DirectionIcon size={14} /> 1D {formatPercent(market?.price_change_pct_1d)}</small>
          </article>
          <article><span>市场时点</span><strong>{formatTime(market?.as_of || dashboard?.as_of)}</strong><small>研究中的所有结论以此为准</small></article>
          <article><span>短期研究基线</span><strong>{forecast?.stance || (state.kind === 'loading' ? '读取中' : '暂无')}</strong><small>{forecast ? `${forecast.confidence_band}置信度 · ${forecast.action}` : '等待有效基线'}</small></article>
          <article><span>数据新鲜度</span><strong>{market ? `${market.freshness_seconds}s` : '—'}</strong><small>{state.kind === 'delayed' ? '已延迟，请降低结论权重' : state.kind === 'degraded' ? '部分来源已降级' : '按数据时点审读'}</small></article>
        </div>
      </DataStateBoundary>
    </section>
  );
}

function NewsList({ items, primary = false }) {
  if (!items.length) return <p className="dashboard-empty">暂无可审计条目。</p>;
  return (
    <div className="source-news-list">
      {items.map((item) => (
        <article key={item.event_id}>
          <div className="source-news-meta">
            <span>{primary ? '官方原始来源' : '补充来源'}</span>
            <small>{item.source_authority || item.source} · {formatTime(item.published_at)}</small>
          </div>
          <h3>{item.title}</h3>
          <p>{item.summary}</p>
          {item.url ? <a href={item.url} target="_blank" rel="noreferrer">打开原文 <ExternalLink size={13} /></a> : null}
        </article>
      ))}
    </div>
  );
}

export function PrimarySourceFeed({ news = [], state }) {
  const primary = news.filter((item) => item.is_primary_source || item.source_tier === 'primary');
  const background = news.filter((item) => !item.is_primary_source && item.source_tier !== 'primary' && item.source_tier !== 'synthetic');
  return (
    <section className="primary-source-section" data-dashboard-section="primary-news" aria-labelledby="primary-news-title">
      <div className="dashboard-section-head">
        <div>
          <span className="section-index">02 · PRIMARY SOURCES</span>
          <h2 id="primary-news-title">一手信息</h2>
          <p>官方发布与媒体解读分层展示，不把转载当作多方共识。</p>
        </div>
        <Newspaper size={20} />
      </div>
      {state.kind === 'unavailable' ? <p className="dashboard-empty">当前市场链路不可用，一手信息未能刷新。</p> : <NewsList items={primary} primary />}
      {background.length ? (
        <details className="background-news">
          <summary>补充背景 <span>{background.length} 条</span></summary>
          <NewsList items={background} />
        </details>
      ) : null}
    </section>
  );
}

export function ResearchSummary({ insights, state }) {
  const thesis = insights?.thesis;
  return (
    <section className="research-summary-section" data-dashboard-section="research-summary" aria-labelledby="research-summary-title">
      <div className="dashboard-section-head">
        <div>
          <span className="section-index">03 · SYNTHESIS</span>
          <h2 id="research-summary-title">今日研究摘要</h2>
          <p>结论、原因、失效条件和复核时点放在同一屏。</p>
        </div>
        <Clock3 size={20} />
      </div>
      {state.kind === 'unavailable' || state.kind === 'loading' ? (
        <p className="dashboard-empty">{state.kind === 'loading' ? '正在生成与当前时点一致的研究摘要。' : '因当前数据不可用，今日研究摘要已暂停。'}</p>
      ) : (
        <div className="research-summary-grid">
          <article><span>核心结论</span><strong>{thesis?.headline || '保持情景研究'}</strong><p>{thesis?.summary}</p></article>
          <article><span>最强支撑</span><strong>{thesis?.support || '待观察'}</strong><p>需要价格与一手信息同时验证。</p></article>
          <article><span>关键失效</span><strong>{thesis?.invalidator || '数据或市场反应反转'}</strong><p>触发后应重新运行研究案件。</p></article>
        </div>
      )}
    </section>
  );
}
