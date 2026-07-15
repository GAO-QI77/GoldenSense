import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Clock3,
  Compass,
  Database,
  Gauge,
  Layers,
  Loader2,
  PieChart,
  Scale,
  Sparkles,
  Target,
  Waves,
} from 'lucide-react';

import { addSearchEntries } from './searchIndex';

const API_URL = import.meta.env.VITE_AGENT_API_URL || '/api/v1/agent/analyze';
const RESEARCH_URL =
  import.meta.env.VITE_AGENT_RESEARCH_URL || API_URL.replace('/analyze', '/research/current');
const CALIBRATION_URL =
  import.meta.env.VITE_AGENT_CALIBRATION_URL || API_URL.replace('/analyze', '/calibration');
const API_KEY = import.meta.env.VITE_AGENT_API_KEY || 'dev-public-key';

const STATE_META = {
  calm: { label: '平静', color: 'var(--green)' },
  elevated: { label: '波动升高', color: 'var(--amber)' },
  stress: { label: '压力', color: 'var(--red)' },
};

const FACTOR_LABELS = {
  real_rate_momentum: { label: '实际利率动量', hint: '10Y TIPS 实际收益率 63 日下行 → 利多' },
  usd_downtrend: { label: '美元趋势', hint: '美元指数低于 50 日均线 → 利多' },
  inflation_expectation: { label: '通胀预期动量', hint: '10Y 盈亏平衡通胀 63 日上行 → 利多' },
  flow_confirmation: { label: '资金流确认(代理)', hint: 'GLD 美元成交量扩张确认价格趋势' },
  price_trend_fallback: { label: '价格趋势(降级)', hint: '宏观列缺失时回退 200 日趋势' },
};

const PROFILE_LABELS = {
  conservative: '保守型',
  balanced: '平衡型',
  aggressive: '进取型',
};

function headers() {
  return { 'Content-Type': 'application/json', 'X-API-Key': API_KEY };
}

async function fetchJson(url, signal, fallbackMessage) {
  const response = await fetch(url, { headers: headers(), signal });
  const text = await response.text();
  let json = null;
  if (text.trim()) {
    try {
      json = JSON.parse(text);
    } catch (error) {
      throw new Error(`${fallbackMessage}：返回非 JSON 内容`);
    }
  }
  if (!response.ok) {
    const detail = json?.detail;
    const message = typeof detail === 'string' ? detail : detail?.message || json?.message;
    throw new Error(message || `${fallbackMessage}：HTTP ${response.status}`);
  }
  return json;
}

const fmt = {
  pct: (x, digits = 1) => (x == null ? '—' : `${(x * 100).toFixed(digits)}%`),
  pctRaw: (x, digits = 1) => (x == null ? '—' : `${Number(x).toFixed(digits)}%`),
  price: (x) => (x == null ? '—' : `$${Number(x).toLocaleString('en-US', { maximumFractionDigits: 0 })}`),
  num: (x, digits = 2) => (x == null ? '—' : Number(x).toFixed(digits)),
};

export default function QuantPage() {
  const [research, setResearch] = useState(null);
  const [calibration, setCalibration] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      setLoading(true);
      setError('');
      try {
        const [researchJson, calibrationJson] = await Promise.all([
          fetchJson(RESEARCH_URL, controller.signal, '量化研究数据读取失败'),
          fetchJson(CALIBRATION_URL, controller.signal, '校准数据读取失败').catch(() => null),
        ]);
        setResearch(researchJson);
        setCalibration(calibrationJson);
      } catch (loadError) {
        if (loadError.name !== 'AbortError') {
          setError(loadError.message || '量化研究数据读取失败');
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    load();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    addSearchEntries('quant', [
      { id: 'q-regime', source: 'quant', title: '市场状态 HMM 后验概率', hint: 'calm / elevated / stress 概率带', route: '/quant', hash: 'panel-regime', keywords: ['regime', 'hmm', '状态', '波动'] },
      { id: 'q-cone', source: 'quant', title: '90 日蒙特卡洛情景锥', hint: 'P10 / P50 / P90 分布路径', route: '/quant', hash: 'panel-cone', keywords: ['monte carlo', '情景', '锥', '预测分布'] },
      { id: 'q-fair', source: 'quant', title: '宏观公允价值锚', hint: '实际利率 + 美元误差修正模型', route: '/quant', hash: 'panel-fair-value', keywords: ['fair value', '公允', '估值', '实际利率'] },
      { id: 'q-factors', source: 'quant', title: '宏观因子面板', hint: '实际利率 / 美元 / 通胀预期 / 资金流', route: '/quant', hash: 'panel-factors', keywords: ['因子', 'factor', '宏观'] },
      { id: 'q-vol', source: 'quant', title: 'HAR-RV 波动率与收益分布带', hint: 'T+1 / T+5 / T+21 P10-P90 区间', route: '/quant', hash: 'panel-vol', keywords: ['波动率', 'har', '分位数', '区间'] },
      { id: 'q-alloc', source: 'quant', title: '战略配置区间 (BL-lite)', hint: '按风险画像的黄金配置建议区间', route: '/quant', hash: 'panel-allocation', keywords: ['配置', 'allocation', 'black litterman'] },
      { id: 'q-calibration', source: 'quant', title: 'Agent 校准记分卡', hint: '历史判断命中率与 Brier 分数', route: '/quant', hash: 'panel-calibration', keywords: ['校准', '命中率', 'brier', 'scorecard'] },
    ]);
  }, []);

  const degraded = research?.degraded || {};

  return (
    <main className="page-surface quant-page">
      <section className="terminal-header quant-header">
        <div>
          <p className="eyebrow">Quant Research Terminal</p>
          <h1>量化研究面板</h1>
          <p>
            长周期本地数据集（{research?.data_span ? `${research.data_span[0]} → ${research.data_span[1]}` : '2004 年起'}
            ，{research?.data_rows ? `${research.data_rows.toLocaleString()} 个交易日` : '…'}）驱动的状态机、估值锚与情景分布。
            所有区块要么给出数据，要么显式标注降级——不伪装。
          </p>
        </div>
        <div className={`status-badge ${loading ? 'is-loading' : error ? 'is-error' : 'is-ok'}`}>
          {loading ? <Loader2 size={15} className="spin" /> : <Sparkles size={15} />}
          {loading ? '加载中' : error ? '数据异常' : '量化链路正常'}
        </div>
      </section>

      {error && (
        <div className="quant-error" role="alert">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {!error && research && <DataFreshnessBanner research={research} />}

      {!error && (
        <div className="quant-grid">
          <RegimePanel regime={research?.regime_posterior} degraded={degraded.regime_posterior} />
          <ConePanel cone={research?.scenario_cone} degraded={degraded.scenario_cone} />
          <FairValuePanel fairValue={research?.fair_value} degraded={degraded.fair_value} />
          <FactorPanel factors={research?.macro_factors} degraded={degraded.macro_factors} />
          <VolPanel volBands={research?.vol_bands} degraded={degraded.vol_bands} />
          <AllocationPanel allocation={research?.allocation} degraded={degraded.allocation} />
          <CalibrationPanel calibration={calibration} />
        </div>
      )}

      <footer className="quant-footnote">
        <Scale size={14} />
        教育型研究辅助，非投资建议。全部模型输出附带方法与失效条件；历史回测不代表未来表现。
      </footer>
    </main>
  );
}

const GOVERNANCE_META = {
  champion: { label: '冠军姿态', tone: 'ok' },
  watch: { label: '观察期', tone: 'warn' },
  demoted: { label: '性能降级', tone: 'bad' },
  insufficient_data: { label: '样本不足', tone: 'muted' },
};

function GovernanceBadge({ governance }) {
  const meta = GOVERNANCE_META[governance.mode] || GOVERNANCE_META.insufficient_data;
  return (
    <div className={`governance-badge ${meta.tone}`} role="status">
      <span className="governance-dot" />
      <div>
        <strong>模型治理：{meta.label}</strong>
        <small>{governance.reason}</small>
      </div>
      {governance.demoted && <span className="governance-flag">已收敛为保守姿态</span>}
    </div>
  );
}

function DataFreshnessBanner({ research }) {
  const asof = research.data_asof;
  const ageDays = research.data_age_days;
  const stale = research.data_stale;
  const source = research.data_source;
  const extendedMissing = Boolean(research.degraded?.extended_dataset);

  const tone = stale || extendedMissing ? 'warn' : 'info';
  const ageText =
    ageDays == null ? '' : ageDays <= 0 ? '当日收盘' : `${ageDays} 天前收盘`;

  return (
    <div className={`data-freshness ${tone}`} role="status">
      <span className="data-freshness-icon">
        {stale || extendedMissing ? <AlertTriangle size={16} /> : <Clock3 size={16} />}
      </span>
      <div className="data-freshness-body">
        <strong>
          研究数据截至 {asof || '—'}
          {ageText && <span className="data-freshness-age"> · {ageText}</span>}
        </strong>
        <small>
          本页由仓库本地长历史数据集（{source === 'extended' ? '扩展 2004+' : '基础样本'}）离线计算，
          <b>非实时行情</b>；现价与首页实时价可能存在时间差。
          {stale && ' 数据已超出新鲜窗口，请以最新行情为准。'}
          {extendedMissing && ' 警告：扩展数据集缺失，量化模型已回退至较短的基础样本。'}
        </small>
      </div>
      <span className="data-freshness-tag">
        <Database size={13} />
        {source === 'extended' ? '扩展数据集' : '基础样本'}
      </span>
    </div>
  );
}

function PanelShell({ id, icon: Icon, title, subtitle, degraded, children, wide }) {
  return (
    <section id={id} className={wide ? 'quant-panel wide' : 'quant-panel'}>
      <header className="quant-panel-head">
        <span className="quant-panel-icon">
          <Icon size={16} />
        </span>
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
      </header>
      {degraded ? (
        <div className="quant-degraded">
          <AlertTriangle size={14} />
          该模块已降级：{String(degraded)}
        </div>
      ) : (
        children
      )}
    </section>
  );
}

/* ------------------------------ regime ---------------------------------- */
function RegimePanel({ regime, degraded }) {
  const latest = regime?.latest;
  const series = regime?.series_tail;
  const dates = regime?.dates_tail || [];

  const chart = useMemo(() => {
    if (!series) return null;
    const keys = ['calm', 'elevated', 'stress'].filter((key) => series[key]);
    const n = series[keys[0]]?.length || 0;
    if (!n) return null;
    const width = 620;
    const height = 150;
    const x = (i) => (i / Math.max(n - 1, 1)) * width;
    const paths = [];
    let baseline = new Array(n).fill(0);
    for (const key of keys) {
      const upper = baseline.map((b, i) => b + series[key][i]);
      const top = upper.map((v, i) => `${x(i).toFixed(1)},${(height - v * height).toFixed(1)}`);
      const bottom = baseline
        .map((v, i) => `${x(i).toFixed(1)},${(height - v * height).toFixed(1)}`)
        .reverse();
      paths.push({ key, d: `M${top.join('L')}L${bottom.join('L')}Z` });
      baseline = upper;
    }
    return { paths, width, height };
  }, [series]);

  return (
    <PanelShell
      id="panel-regime"
      icon={Waves}
      title="市场状态 · HMM 后验概率"
      subtitle="三状态高斯隐马尔可夫模型，日收益 + 已实现波动率拟合；概率加权驱动暴露折扣"
      degraded={degraded}
      wide
    >
      {latest && (
        <div className="regime-latest">
          {Object.entries(STATE_META).map(([key, meta]) => (
            <div key={key} className="regime-chip" style={{ '--chip-color': meta.color }}>
              <span className="regime-dot" />
              <span>{meta.label}</span>
              <strong>{fmt.pct(latest[key] ?? 0, 0)}</strong>
            </div>
          ))}
        </div>
      )}
      {chart && (
        <figure className="quant-chart">
          <svg viewBox={`0 0 ${chart.width} ${chart.height}`} role="img" aria-label="近一年市场状态概率堆叠图">
            {chart.paths.map(({ key, d }) => (
              <path key={key} d={d} fill={STATE_META[key].color} opacity="0.75" />
            ))}
          </svg>
          <figcaption>
            近 {dates.length} 个交易日（{dates[0]} → {dates[dates.length - 1]}）的状态概率堆叠；
            由本地长历史拟合，非实时行情。
          </figcaption>
        </figure>
      )}
    </PanelShell>
  );
}

/* ------------------------------- cone ----------------------------------- */
function ConePanel({ cone, degraded }) {
  const chart = useMemo(() => {
    if (!cone?.percentile_paths) return null;
    const { p10, p50, p90 } = cone.percentile_paths;
    const n = p50.length;
    const width = 620;
    const height = 190;
    const values = [...p10, ...p90, cone.spot];
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const pad = (hi - lo) * 0.06 || 1;
    const y = (v) => height - ((v - (lo - pad)) / (hi - lo + 2 * pad)) * height;
    const x = (i) => (i / Math.max(n - 1, 1)) * width;
    const line = (arr) => arr.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('L');
    const band = `M${line(p90)}L${p10
      .map((v, i) => `${x(n - 1 - i).toFixed(1)},${y(p10[n - 1 - i]).toFixed(1)}`)
      .join('L')}Z`;
    return { width, height, band, median: `M${line(p50)}`, spotY: y(cone.spot) };
  }, [cone]);

  const d30 = cone?.checkpoints?.d30;
  const d90 = cone?.checkpoints?.d90;

  return (
    <PanelShell
      id="panel-cone"
      icon={Compass}
      title="90 日情景锥 · 蒙特卡洛"
      subtitle={`基于 HMM 状态转移的 ${cone?.n_paths?.toLocaleString() || '—'} 条模拟路径；分布代替点预测`}
      degraded={degraded}
      wide
    >
      {chart && (
        <figure className="quant-chart">
          <svg viewBox={`0 0 ${chart.width} ${chart.height}`} role="img" aria-label="90日价格情景锥">
            <path d={chart.band} fill="var(--gold)" opacity="0.18" />
            <path d={chart.median} fill="none" stroke="var(--gold)" strokeWidth="2" />
            <line
              x1="0"
              x2={chart.width}
              y1={chart.spotY}
              y2={chart.spotY}
              stroke="var(--carbon-muted)"
              strokeDasharray="5 5"
              strokeWidth="1"
            />
          </svg>
          <figcaption>
            金色带为 P10–P90 区间，实线为中位路径，虚线为现价 {fmt.price(cone.spot)}。
          </figcaption>
        </figure>
      )}
      <div className="cone-checkpoints">
        {[['T+30', d30], ['T+90', d90]].map(([label, cp]) => (
          <div key={label} className="cone-checkpoint">
            <span className="cone-label">{label}</span>
            {cp ? (
              <>
                <strong>
                  {fmt.price(cp.p10)} ~ {fmt.price(cp.p90)}
                </strong>
                <small>
                  中位 {fmt.price(cp.p50)} · 高于现价概率 {fmt.pct(cp.prob_above_spot, 0)}
                </small>
              </>
            ) : (
              <small>—</small>
            )}
          </div>
        ))}
      </div>
    </PanelShell>
  );
}

/* ----------------------------- fair value -------------------------------- */
function FairValuePanel({ fairValue, degraded }) {
  const deviation = fairValue?.deviation_pct;
  const tail = fairValue?.deviation_series_tail || [];
  const spark = useMemo(() => {
    if (tail.length < 2) return null;
    const width = 260;
    const height = 56;
    const lo = Math.min(...tail, 0);
    const hi = Math.max(...tail, 0);
    const y = (v) => height - ((v - lo) / (hi - lo || 1)) * height;
    const x = (i) => (i / (tail.length - 1)) * width;
    return {
      width,
      height,
      zeroY: y(0),
      d: `M${tail.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('L')}`,
    };
  }, [tail]);

  const quartiles = fairValue?.quartile_forward_returns || {};

  return (
    <PanelShell
      id="panel-fair-value"
      icon={Gauge}
      title="宏观公允价值锚"
      subtitle="log(金价) ~ 实际利率 + 美元指数，滚动十年窗口误差修正框架"
      degraded={degraded}
    >
      {(() => {
        const regimeBreak = Boolean(fairValue?.regime_break);
        const tone = regimeBreak ? 'break' : deviation >= 0 ? 'rich' : 'cheap';
        const label = regimeBreak
          ? '结构性偏离期'
          : `相对宏观公允值（${deviation >= 0 ? '偏贵' : '偏便宜'}）`;
        return (
          <div className="fair-value-hero">
            <div>
              <span className="fair-value-figure" data-tone={tone}>
                {deviation == null ? '—' : `${deviation >= 0 ? '+' : ''}${fmt.num(deviation, 1)}%`}
              </span>
              <small>
                {label}
                {fairValue?.deviation_z != null && (
                  <span className="fair-value-z"> · {fmt.num(Math.abs(fairValue.deviation_z), 1)}σ</span>
                )}
              </small>
            </div>
            <dl className="fair-value-meta">
              <div>
                <dt>模型公允值</dt>
                <dd>{fmt.price(fairValue?.fair_value)}</dd>
              </div>
              <div>
                <dt>现价</dt>
                <dd>{fmt.price(fairValue?.spot)}</dd>
              </div>
              <div>
                <dt>拟合 R²</dt>
                <dd>{fmt.num(fairValue?.r_squared, 2)}</dd>
              </div>
              <div>
                <dt>误差修正半衰期</dt>
                <dd>{fairValue?.half_life_days ? `${Math.round(fairValue.half_life_days)} 日` : '未见回归'}</dd>
              </div>
            </dl>
          </div>
        );
      })()}

      {fairValue?.interpretation && (
        <p className={`fair-value-interpretation${fairValue?.regime_break ? ' is-break' : ''}`}>
          {fairValue?.regime_break && <AlertTriangle size={14} />}
          {fairValue.interpretation}
        </p>
      )}
      {spark && (
        <figure className="quant-chart small">
          <svg viewBox={`0 0 ${spark.width} ${spark.height}`} role="img" aria-label="近一年估值偏离走势">
            <line x1="0" x2={spark.width} y1={spark.zeroY} y2={spark.zeroY} stroke="var(--line-strong)" strokeWidth="1" />
            <path d={spark.d} fill="none" stroke="var(--gold)" strokeWidth="1.8" />
          </svg>
          <figcaption>近 260 个交易日估值偏离（%）</figcaption>
        </figure>
      )}
      <div className="quartile-card">
        <h3>历史证据：按偏离四分位的 12 个月远期回报均值</h3>
        <div className="quartile-row">
          {['q1_cheapest', 'q2', 'q3', 'q4_richest'].map((key) => (
            <div key={key} className="quartile-cell">
              <small>{{ q1_cheapest: '最便宜', q2: 'Q2', q3: 'Q3', q4_richest: '最贵' }[key]}</small>
              <strong>{quartiles[key] != null ? fmt.pct(quartiles[key], 1) : '—'}</strong>
            </div>
          ))}
        </div>
        <p className="quant-note">注意：2022 年后央行购金抬升了结构性需求，估值锚采用滚动窗口以适应机制变化。</p>
      </div>
    </PanelShell>
  );
}

/* ------------------------------ factors ---------------------------------- */
function FactorPanel({ factors, degraded }) {
  const latest = factors?.latest || {};
  const used = factors?.factors_used || [];
  return (
    <PanelShell
      id="panel-factors"
      icon={Layers}
      title="宏观因子面板"
      subtitle={`中期组合的四个先验因子；当前组合得分 ${factors ? fmt.num(factors.composite, 2) : '—'} / 1.00`}
      degraded={degraded}
    >
      <ul className="factor-list">
        {used.map((key) => {
          const meta = FACTOR_LABELS[key] || { label: key, hint: '' };
          const on = (latest[key] ?? 0) >= 0.5;
          return (
            <li key={key} className={on ? 'factor-item on' : 'factor-item'}>
              <span className="factor-light" aria-hidden="true" />
              <div>
                <strong>{meta.label}</strong>
                <small>{meta.hint}</small>
              </div>
              <span className="factor-state">{on ? '利多' : '未触发'}</span>
            </li>
          );
        })}
      </ul>
      <p className="quant-note">
        因子参数全部先验设定（未对本数据拟合），并经 2004 年以来含成本回测评估；资金流为成交额代理。
      </p>
    </PanelShell>
  );
}

/* -------------------------------- vol ------------------------------------ */
function VolPanel({ volBands, degraded }) {
  const rows = [
    ['h1', 'T+1'],
    ['h5', 'T+5'],
    ['h21', 'T+21'],
  ].filter(([key]) => volBands?.[key]);

  const bounds = useMemo(() => {
    if (!rows.length) return null;
    const values = rows.flatMap(([key]) => [volBands[key].p10, volBands[key].p90]);
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    return { lo, hi, span: hi - lo || 1 };
  }, [rows, volBands]);

  return (
    <PanelShell
      id="panel-vol"
      icon={Activity}
      title="波动率与收益分布带"
      subtitle={`HAR-RV 波动率预测 + 经验分位数；年化波动预测 ${volBands?.h1 ? fmt.pct(volBands.h1.ann_vol_forecast, 1) : '—'}`}
      degraded={degraded}
    >
      <div className="vol-band-table">
        {rows.map(([key, label]) => {
          const band = volBands[key];
          const left = ((band.p10 - bounds.lo) / bounds.span) * 100;
          const width = ((band.p90 - band.p10) / bounds.span) * 100;
          const median = ((band.p50 - bounds.lo) / bounds.span) * 100;
          return (
            <div key={key} className="vol-band-row">
              <span className="vol-band-label">{label}</span>
              <div className="vol-band-track">
                <span className="vol-band-range" style={{ left: `${left}%`, width: `${width}%` }} />
                <span className="vol-band-median" style={{ left: `${median}%` }} />
              </div>
              <span className="vol-band-figures">
                {fmt.pct(band.p10, 1)} / {fmt.pct(band.p50, 1)} / {fmt.pct(band.p90, 1)}
              </span>
            </div>
          );
        })}
      </div>
      <p className="quant-note">
        短期层不预测方向（已被走前验证证伪），改为输出可校验的收益分布：P10 / P50 / P90。
      </p>
    </PanelShell>
  );
}

/* ----------------------------- allocation -------------------------------- */
function AllocationPanel({ allocation, degraded }) {
  const profiles = ['conservative', 'balanced', 'aggressive'].filter((p) => allocation?.[p]);
  const maxPct = 25;
  return (
    <PanelShell
      id="panel-allocation"
      icon={PieChart}
      title="战略配置参考区间 · BL-lite"
      subtitle="风险画像先验 × 状态观点 × 估值观点（观点只倾斜先验，上限 ±25%）· 研究参考，非投资建议"
      degraded={degraded}
    >
      <div className="alloc-table">
        {profiles.map((profile) => {
          const advice = allocation[profile];
          const [pLo, pHi] = advice.prior_range_pct;
          const [rLo, rHi] = advice.reference_range_pct || advice.recommended_range_pct;
          return (
            <div key={profile} className="alloc-row">
              <span className="alloc-label">{PROFILE_LABELS[profile]}</span>
              <div className="alloc-track">
                <span
                  className="alloc-prior"
                  style={{ left: `${(pLo / maxPct) * 100}%`, width: `${((pHi - pLo) / maxPct) * 100}%` }}
                />
                <span
                  className="alloc-recommended"
                  style={{ left: `${(rLo / maxPct) * 100}%`, width: `${(Math.max(rHi - rLo, 0.4) / maxPct) * 100}%` }}
                />
              </div>
              <span className="alloc-figures">
                {fmt.pctRaw(rLo)} – {fmt.pctRaw(rHi)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="alloc-legend">
        <span><i className="legend-prior" />画像先验区间</span>
        <span><i className="legend-recommended" />观点倾斜后参考区间</span>
      </div>
      {allocation?.balanced?.disclaimer && (
        <p className="alloc-disclaimer">{allocation.balanced.disclaimer}</p>
      )}
      {allocation?.balanced?.rationale && (
        <ul className="alloc-rationale">
          {allocation.balanced.rationale.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      )}
    </PanelShell>
  );
}

/* ---------------------------- calibration -------------------------------- */
function CalibrationPanel({ calibration }) {
  const hasData = calibration && calibration.total_scored > 0;
  return (
    <PanelShell
      id="panel-calibration"
      icon={Target}
      title="Agent 校准记分卡"
      subtitle="系统回头给自己的历史判断打分：命中率、Brier 分数与最近判定"
      degraded={calibration ? null : '校准服务暂不可用'}
      wide
    >
      {!hasData && (
        <p className="quant-note">
          尚无足够的已到期历史判断可供评分。每次分析都会入库，24h / 7d / 30d 到期后自动回填实际金价走势并计分。
        </p>
      )}
      {calibration?.governance && <GovernanceBadge governance={calibration.governance} />}
      {hasData && (
        <>
          <div className="calib-tiles">
            <div className="calib-tile">
              <small>已评分判断</small>
              <strong>{calibration.total_scored}</strong>
            </div>
            <div className="calib-tile">
              <small>方向性判断命中率</small>
              <strong>{calibration.hit_rate != null ? fmt.pct(calibration.hit_rate, 0) : '—'}</strong>
            </div>
            <div className="calib-tile">
              <small>Brier 分数（越低越好）</small>
              <strong>{calibration.brier_score != null ? fmt.num(calibration.brier_score, 3) : '—'}</strong>
            </div>
            <div className="calib-tile">
              <small>观望/风险门控占比</small>
              <strong>
                {fmt.pct(
                  calibration.total_scored
                    ? calibration.neutral_or_gated / calibration.total_scored
                    : null,
                  0,
                )}
              </strong>
            </div>
          </div>
          {calibration.recent_outcomes?.length > 0 && (
            <div className="calib-list">
              {calibration.recent_outcomes.slice(-8).reverse().map((o) => (
                <div key={o.analysis_id} className="calib-item">
                  <span className={`calib-hit ${o.hit === true ? 'hit' : o.hit === false ? 'miss' : ''}`}>
                    {o.hit === true ? '命中' : o.hit === false ? '未中' : '观望'}
                  </span>
                  <span className="calib-stance">{o.stance} · {o.horizon}</span>
                  <span className="calib-return">{fmt.pct(o.realized_return, 2)}</span>
                  <span className="calib-date">{String(o.created_at).slice(0, 10)}</span>
                </div>
              ))}
            </div>
          )}
          <p className="quant-note">
            校准结果以硬上限（±20%）反哺委员会置信度：{calibration.weight_adjustment?.basis || '—'}。
          </p>
        </>
      )}
    </PanelShell>
  );
}
