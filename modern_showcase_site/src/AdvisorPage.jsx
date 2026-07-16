import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  FileSearch,
  Loader2,
  ShieldCheck,
  Sparkles,
  SlidersHorizontal,
  User,
  Wand2,
} from 'lucide-react';

import {
  AdvancedProfileFields,
  CoreProfileFields,
} from './profileFields';
import { loadProfile, saveProfile, toPersonalResearchBody } from './profileStore';

const API_URL = import.meta.env.VITE_AGENT_API_URL || '/api/v1/agent/analyze';
const PERSONAL_URL =
  import.meta.env.VITE_AGENT_PERSONAL_URL || API_URL.replace('/analyze', '/personal-research');
const API_KEY = import.meta.env.VITE_AGENT_API_KEY || 'dev-public-key';

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
  drawdown_tolerance_mismatch: '回撤承受力可能被击穿',
  leverage_out_of_scope: '杠杆超出研究口径',
  liquidity_horizon_mismatch: '流动性与期限矛盾',
};

async function postPersonal(body, mode, signal) {
  const response = await fetch(`${PERSONAL_URL}?mode=${mode}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-API-Key': API_KEY },
    body: JSON.stringify(body),
    signal,
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
  return json;
}

export default function AdvisorPage() {
  const [profile, setProfile] = useState(loadProfile);
  const [result, setResult] = useState(null);
  const [phase, setPhase] = useState('idle'); // idle | draft-loading | polishing | done | draft-only
  const [error, setError] = useState('');
  const generationRef = useRef(0);

  useEffect(() => {
    saveProfile(profile);
  }, [profile]);

  function update(key, value) {
    setProfile((current) => ({ ...current, [key]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const generation = ++generationRef.current;
    const body = toPersonalResearchBody(profile);
    setError('');
    setPhase('draft-loading');

    // Phase 1: deterministic draft — every number, in about a second.
    try {
      const draft = await postPersonal(body, 'draft');
      if (generationRef.current !== generation) return;
      setResult(draft);
      setPhase('polishing');
    } catch (draftError) {
      if (generationRef.current !== generation) return;
      setError(draftError.message || '个性化研究生成失败');
      setPhase('idle');
      return;
    }

    // Phase 2: LLM polish — swap the narrative in place when it lands.
    try {
      const full = await postPersonal(body, 'full');
      if (generationRef.current !== generation) return;
      setResult(full);
      setPhase('done');
    } catch {
      if (generationRef.current !== generation) return;
      // Draft stays on screen; numbers are identical by construction.
      setPhase('draft-only');
    }
  }

  const facts = result?.facts;
  const narrative = result?.narrative;
  const gap = facts?.position_gap;
  const range = facts?.reference_range;
  const gapMeta = gapLabels[gap?.status] || gapLabels.unknown;
  const horizonSection = facts?.horizon_evidence?.section;
  const polishing = phase === 'polishing';
  const loading = phase === 'draft-loading';

  const narrativeStatus = useMemo(() => {
    if (polishing) return { label: 'DeepSeek 正在润色语言…（数字已定稿，不会改变）', tone: 'pending' };
    if (phase === 'draft-only')
      return { label: '本次使用确定性草稿（语言润色暂不可用，数字口径完全一致）', tone: 'muted' };
    if (!result) return { label: '', tone: 'muted' };
    if (result.generated_by === 'llm')
      return { label: 'DeepSeek 叙事 · 已通过数字校验与去指令化双重把关', tone: 'ok' };
    return {
      label: result.degradation_flags?.length
        ? `已回退确定性草稿（${result.degradation_flags.join(' · ')}）`
        : '确定性草稿',
      tone: 'muted',
    };
  }, [phase, polishing, result]);

  return (
    <main className="page-surface advisor-page">
      <section className="terminal-header">
        <div>
          <p className="eyebrow">Personalized Research</p>
          <h1>个性化研究分析</h1>
          <p>
            画像只在本机浏览器保存并随请求发送，服务端不存储；与「风险画像 Agent」页共用同一份画像。
            输出恒为「参考区间 / 差距 / 风险提示」研究框架——不是操作指令。
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
              <span>核心 4 项必填 · 进阶可选 · 全站共享</span>
            </div>
          </div>

          <CoreProfileFields profile={profile} onChange={update} />
          <AdvancedProfileFields profile={profile} onChange={update} />

          {error ? (
            <div className="error-panel">
              <AlertTriangle size={18} />
              <div>
                <strong>生成失败</strong>
                <p>{error}</p>
              </div>
            </div>
          ) : null}

          <button className="primary-action" type="submit" disabled={loading || polishing}>
            {loading ? <Loader2 size={17} className="spinning" /> : <Wand2 size={17} />}
            {loading ? '计算数字中（约 1 秒）' : polishing ? '数字已出 · 语言润色中' : '生成个性化研究分析'}
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
            <li>提交后约 1 秒先看到全部数字（规则引擎确定性产出），DeepSeek 随后仅替换语言表述。</li>
            <li>每个数字经叙事校验器逐一核对，未着地则整体回退确定性草稿。</li>
            <li>出现任何指令式措辞（「建议买入」等）同样触发回退——本页永不输出买卖指令。</li>
            <li>进阶画像可选：填写回撤承受力/流动性/杠杆态度后，解锁对应的错配检查规则。</li>
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
              <p className={`narrative-status ${narrativeStatus.tone}`}>
                {polishing ? <Loader2 size={13} className="spinning" /> : null}
                {narrativeStatus.tone === 'ok' ? <Sparkles size={13} /> : null}
                {narrativeStatus.label}
              </p>
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

            <section className={`panel-block ${polishing ? 'narrative-polishing' : ''}`}>
              <div className="panel-title">
                <BrainCircuit size={16} />
                <div>
                  <h2>定制叙事</h2>
                  <span>
                    {polishing
                      ? '润色中 · 以下为确定性草稿'
                      : result.degradation_flags?.length
                        ? result.degradation_flags.join(' · ')
                        : '双重把关通过'}
                  </span>
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
        <span>数字先行 · 语言后补 · 双重把关 · 画像不落库</span>
      </footer>
    </main>
  );
}
