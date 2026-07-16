import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  FileSearch,
  Loader2,
  ShieldCheck,
  SlidersHorizontal,
  User,
  Wand2,
} from 'lucide-react';

const API_URL = import.meta.env.VITE_AGENT_API_URL || '/api/v1/agent/analyze';
const PERSONAL_URL =
  import.meta.env.VITE_AGENT_PERSONAL_URL || API_URL.replace('/analyze', '/personal-research');
const API_KEY = import.meta.env.VITE_AGENT_API_KEY || 'dev-public-key';

const STORAGE_KEY = 'gs_advisor_profile_v1';

const defaultProfile = {
  risk_tolerance: 'balanced',
  horizon: 'mid',
  current_gold_pct: 10,
  experience: 'novice',
};

const riskOptions = [
  ['conservative', '保守型'],
  ['balanced', '稳健型'],
  ['aggressive', '进取型'],
];
const horizonOptions = [
  ['short', '短期(1-21天)'],
  ['mid', '中期(1-6月)'],
  ['long', '长期(6月+)'],
];
const experienceOptions = [
  ['novice', '新手'],
  ['experienced', '有经验'],
  ['professional', '专业'],
];

const gapLabels = {
  within: { text: '处于参考区间内', tone: 'bull' },
  above: { text: '高于参考区间上沿', tone: 'risk' },
  below: { text: '低于参考区间下沿', tone: 'neutral' },
  unknown: { text: '区间不可用', tone: 'neutral' },
};

const flagLabels = {
  position_above_range_in_stress: '压力状态下仓位超区间',
  position_far_above_range: '仓位显著高于区间',
  short_horizon_high_vol: '短期限 × 高波动',
  structural_valuation_deviation: '估值结构性偏离',
  stale_data: '数据超出新鲜度阈值',
};

function loadStoredProfile() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultProfile;
    const parsed = JSON.parse(raw);
    return { ...defaultProfile, ...parsed };
  } catch {
    return defaultProfile;
  }
}

export default function AdvisorPage() {
  const [profile, setProfile] = useState(loadStoredProfile);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(profile));
    } catch {
      /* localStorage unavailable: profile stays session-only */
    }
  }, [profile]);

  function update(key, value) {
    setProfile((current) => ({ ...current, [key]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError('');
    try {
      const response = await fetch(PERSONAL_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
        body: JSON.stringify({
          ...profile,
          current_gold_pct: Number(profile.current_gold_pct),
        }),
      });
      const text = await response.text();
      const json = text ? JSON.parse(text) : null;
      if (!response.ok) {
        const detail = json?.detail;
        throw new Error(
          (typeof detail === 'string' ? detail : detail?.message) ||
            `个性化研究生成失败：HTTP ${response.status}`,
        );
      }
      setResult(json);
    } catch (submitError) {
      setError(submitError.message || '个性化研究生成失败');
    } finally {
      setLoading(false);
    }
  }

  const facts = result?.facts;
  const narrative = result?.narrative;
  const gap = facts?.position_gap;
  const range = facts?.reference_range;
  const gapMeta = gapLabels[gap?.status] || gapLabels.unknown;
  const horizonSection = facts?.horizon_evidence?.section;
  const generatedByLabel = useMemo(() => {
    if (!result) return '';
    return result.generated_by === 'llm'
      ? 'DeepSeek 叙事 · 已通过数字校验与去指令化双重把关'
      : '确定性草稿（LLM 未启用或被把关回退，数字口径不变）';
  }, [result]);

  return (
    <main className="page-surface advisor-page">
      <section className="terminal-header">
        <div>
          <p className="eyebrow">Personalized Research</p>
          <h1>个性化研究分析</h1>
          <p>
            画像只在本机浏览器保存并随请求发送，服务端不存储。输出恒为「参考区间 / 差距 /
            风险提示」研究框架——不是操作指令。
          </p>
        </div>
        <div className="compliance-pill">
          <ShieldCheck size={15} />
          画像不落库 · 非投顾
        </div>
      </section>

      <section className="advisor-layout">
        <form className="agent-form" onSubmit={handleSubmit}>
          <div className="panel-title">
            <SlidersHorizontal size={16} />
            <div>
              <h2>投资者画像</h2>
              <span>四个字段，保存在 localStorage</span>
            </div>
          </div>

          <SegmentedRow
            label="风险承受能力"
            value={profile.risk_tolerance}
            options={riskOptions}
            onChange={(v) => update('risk_tolerance', v)}
          />
          <SegmentedRow
            label="投资期限"
            value={profile.horizon}
            options={horizonOptions}
            onChange={(v) => update('horizon', v)}
          />
          <SegmentedRow
            label="投资经验"
            value={profile.experience}
            options={experienceOptions}
            onChange={(v) => update('experience', v)}
          />

          <label className="field compact-field">
            <span>当前黄金仓位（占组合 %）</span>
            <div className="number-input">
              <input
                type="number"
                min="0"
                max="100"
                step="0.5"
                value={profile.current_gold_pct}
                onChange={(event) => update('current_gold_pct', event.target.value)}
              />
              <small>%</small>
            </div>
          </label>

          {error ? (
            <div className="error-panel">
              <AlertTriangle size={18} />
              <div>
                <strong>生成失败</strong>
                <p>{error}</p>
              </div>
            </div>
          ) : null}

          <button className="primary-action" type="submit" disabled={loading}>
            {loading ? <Loader2 size={17} className="spinning" /> : <Wand2 size={17} />}
            {loading ? '生成研究分析中' : '生成个性化研究分析'}
          </button>
        </form>

        <aside className="agent-context">
          <div className="panel-title">
            <User size={16} />
            <div>
              <h2>这一页如何工作</h2>
              <span>数字与语言严格分离</span>
            </div>
          </div>
          <ul className="boundary-list">
            <li>全部数字由确定性规则引擎产出：参考区间来自已验证的配置模型，差距与风险旗标为可审计规则。</li>
            <li>DeepSeek 只重写语言；每个数字经叙事校验器逐一核对，未着地则整体回退确定性草稿。</li>
            <li>出现任何指令式措辞（「建议买入」等）同样触发回退——本页永不输出买卖指令。</li>
            <li>画像仅保存在你的浏览器，请求处理完即弃。</li>
          </ul>
        </aside>
      </section>

      {result ? (
        <section className="analysis-output">
          <div className="section-head">
            <div>
              <span>Personalized Briefing</span>
              <h2>你的研究参考</h2>
              <p>{generatedByLabel}</p>
            </div>
          </div>

          <div className="advisor-result-grid">
            <section className={`panel-block advisor-range-panel tone-${gapMeta.tone}`}>
              <div className="panel-title">
                <FileSearch size={16} />
                <div>
                  <h2>参考区间与仓位差距</h2>
                  <span>{range?.available ? range.evidence_ref : '配置区间降级'}</span>
                </div>
              </div>
              {range?.available ? (
                <>
                  <div className="advisor-range-line">
                    <strong>
                      {range.range_pct[0].toFixed(1)}% – {range.range_pct[1].toFixed(1)}%
                    </strong>
                    <span>中点 {range.midpoint}%</span>
                  </div>
                  <div className="advisor-gap-line">
                    <span className={`stance-badge tone-${gapMeta.tone === 'bull' ? 'bull' : gapMeta.tone === 'risk' ? 'risk' : 'neutral'}`}>
                      {gapMeta.text}
                    </span>
                    <span>
                      当前仓位 {gap?.current_gold_pct}%
                      {gap?.gap_pct ? ` · 差距 ${gap.gap_pct} 个百分点` : ''}
                    </span>
                  </div>
                </>
              ) : (
                <p className="muted-copy">配置区间数据降级，本次不提供仓位对比。</p>
              )}
              <div className="flag-stack">
                {(facts?.risk_flags || []).map((flag) => (
                  <span key={flag.flag} title={flag.detail}>
                    {flagLabels[flag.flag] || flag.flag}
                  </span>
                ))}
                {!(facts?.risk_flags || []).length ? (
                  <span className="flag-ok">
                    <CheckCircle2 size={13} /> 未触发风险提示规则
                  </span>
                ) : null}
              </div>
            </section>

            <section className="panel-block">
              <div className="panel-title">
                <BrainCircuit size={16} />
                <div>
                  <h2>定制叙事</h2>
                  <span>{result.degradation_flags?.length ? result.degradation_flags.join(' · ') : '双重把关通过'}</span>
                </div>
              </div>
              <div className="advisor-narrative">
                <p>{narrative?.overview}</p>
                <p>{narrative?.position_analysis}</p>
                {(narrative?.risk_notes || []).map((note) => (
                  <p key={note} className="advisor-risk-note">
                    <AlertTriangle size={13} /> {note}
                  </p>
                ))}
                <p>{narrative?.horizon_note}</p>
              </div>
              {horizonSection?.available ? (
                <div className="advisor-horizon-evidence">
                  <strong>期限匹配证据（{facts.horizon_evidence.horizon}）</strong>
                  <ul>
                    {(horizonSection.evidence || []).slice(0, 3).map((line) => (
                      <li key={line}>{line}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
              <p className="disclaimer">{narrative?.disclaimer}</p>
            </section>
          </div>
        </section>
      ) : null}

      <footer className="terminal-footer">
        <span>GoldenSense Personalized Research</span>
        <span>数字来自规则引擎 · 语言经双重把关 · 画像不落库</span>
      </footer>
    </main>
  );
}

function SegmentedRow({ label, value, options, onChange }) {
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
