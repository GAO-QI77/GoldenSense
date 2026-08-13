import React, { useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  FileSearch,
  Loader2,
  Sparkles,
  Wand2,
} from 'lucide-react';

import { toPersonalResearchBody } from './profileStore';
import { ThreeDimensionalBrief } from './ResearchCasePanel';
import { getResearchHeaders, useActiveResearchCase } from './researchCaseStore';

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
  short_horizon_high_vol: '短期 × 高波动',
  structural_valuation_deviation: '估值结构性偏离',
  stale_data: '数据超出新鲜度阈值',
  drawdown_tolerance_mismatch: '回撤承受力可能被击穿',
  leverage_out_of_scope: '杠杆超出研究口径',
  liquidity_horizon_mismatch: '流动性与期限矛盾',
};

const horizonSectionLabels = {
  short_term: '短期观点',
  mid_term: '中期观点',
  long_term: '长期观点',
};

async function postPersonal(body, mode, signal, caseId = null) {
  const params = new URLSearchParams({ mode });
  if (caseId) params.set('research_case_id', caseId);
  const response = await fetch(`${PERSONAL_URL}?${params.toString()}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getResearchHeaders(API_KEY) },
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

// Capability A of the unified workbench: profile -> reference allocation
// range / position gap / risk flags / two-phase DeepSeek narrative. The
// profile is owned by the parent workbench and passed in; this component only
// renders the "generate" action and its output.
export default function AllocationResearchPanel({ profile }) {
  const activeCase = useActiveResearchCase();
  const [result, setResult] = useState(null);
  const [phase, setPhase] = useState('idle'); // idle | draft-loading | polishing | done | draft-only
  const [error, setError] = useState('');
  const generationRef = useRef(0);

  async function handleSubmit() {
    const generation = ++generationRef.current;
    const body = toPersonalResearchBody(profile);
    setError('');
    setPhase('draft-loading');

    // Phase 1: deterministic draft — every number, in about a second.
    try {
      const draft = await postPersonal(body, 'draft', undefined, activeCase?.case_id);
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
      const full = await postPersonal(body, 'full', undefined, activeCase?.case_id);
      if (generationRef.current !== generation) return;
      setResult(full);
      setPhase('done');
    } catch {
      if (generationRef.current !== generation) return;
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
    <div className="capability-panel">
      <div className="capability-intro">
        <p>
          按你的画像给出<strong>研究口径的参考配置区间</strong>：当前仓位与区间的差距、结构化风险提示，
          以及一段个性化叙事。数字由规则引擎确定性产出、约 1 秒即出；DeepSeek 随后仅润色语言，
          经数字校验与去指令化双重把关。<strong>永不输出买卖指令。</strong>
        </p>
        <button
          className="primary-action"
          type="button"
          onClick={handleSubmit}
          disabled={loading || polishing}
        >
          {loading ? <Loader2 size={17} className="spinning" /> : <Wand2 size={17} />}
          {loading ? '计算数字中（约 1 秒）' : polishing ? '数字已出 · 语言润色中' : '生成个性化配置研究'}
        </button>
        {error ? (
          <div className="error-panel">
            <AlertTriangle size={18} />
            <div>
              <strong>生成失败</strong>
              <p>{error}</p>
            </div>
          </div>
        ) : null}
      </div>

      {result ? (
        <div className="analysis-output">
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
                  <span>研究口径 · 组合占比</span>
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
                  <strong>
                    {horizonSectionLabels[facts.horizon_evidence.horizon] || '期限匹配证据'}
                  </strong>
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
          <ThreeDimensionalBrief brief={result.three_dimensional_brief} />
        </div>
      ) : (
        phase === 'idle' ? (
          <div className="capability-empty">
            <FileSearch size={22} />
            <p>填好上方画像后点击「生成个性化配置研究」，这里会展示参考区间、仓位差距与定制叙事。</p>
          </div>
        ) : null
      )}
    </div>
  );
}
