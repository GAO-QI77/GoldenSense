import React, { useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  DatabaseZap,
  FileImage,
  FileText,
  Fingerprint,
  Globe2,
  Layers3,
  Loader2,
  Radar,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';

import { getResearchHeaders, saveActiveCase, useActiveResearchCase } from './researchCaseStore';

const API_URL = import.meta.env.VITE_AGENT_API_URL || '/api/v1/agent/analyze';
const CASES_URL =
  import.meta.env.VITE_AGENT_RESEARCH_CASES_URL || API_URL.replace('/analyze', '/research-cases');
const API_KEY = import.meta.env.VITE_AGENT_API_KEY || 'dev-public-key';

const gateLabels = {
  access: '接入安全门',
  provenance_time: '来源与时间门',
  fact_location: '事实定位门',
  ai_attack: 'AI攻击门',
  dedup_replay: '去重与重放门',
  consistency: '冲突一致性门',
  market_coherence: '市场一致性门',
  output_audit: '输出审计门',
};

const statusLabels = {
  complete: '研究完成',
  degraded: '显式降级',
  blocked: '证据阻断',
  researching: '研究中',
  created: '已创建',
};

const modelStateLabels = {
  production: '生产',
  watch: '观察',
  degraded: '降级',
  retired: '淘汰',
};

const horizonLabels = {
  short_term: '短期 · 1–21天',
  mid_term: '中期 · 1–6月',
  long_term: '长期 · 6月以上',
};
const agentLabels = {
  macro_event: '事件宏观 Agent',
  technical_flows: '技术与资金流 Agent',
  long_term_fundamental: '长期基本面 Agent',
  quant_model_risk: '量化与模型风险 Agent',
  strategy_arbitrator: '策略仲裁 Agent',
};

async function readJson(response) {
  const text = await response.text();
  let json = null;
  if (text) {
    try {
      json = JSON.parse(text);
    } catch {
      throw new Error('研究服务返回了不可解析内容。');
    }
  }
  if (!response.ok) {
    const detail = json?.detail;
    throw new Error(
      (typeof detail === 'string' ? detail : detail?.message) || `研究案件创建失败：HTTP ${response.status}`,
    );
  }
  return json;
}

export function ResearchCaseWorkspace() {
  const activeCase = useActiveResearchCase();
  const [inputType, setInputType] = useState('url');
  const [question, setQuestion] = useState('这份信息如何影响黄金的短、中、长期研究观点？');
  const [url, setUrl] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileRef = useRef(null);

  async function submit(event) {
    event.preventDefault();
    setError('');
    if (!question.trim()) {
      setError('请先填写研究问题。');
      return;
    }
    if (inputType === 'url' && !url.trim()) {
      setError('请输入HTTPS公开来源URL。');
      return;
    }
    if (inputType !== 'url' && !file) {
      setError(`请选择${inputType === 'pdf' ? 'PDF' : '图片'}文件。`);
      return;
    }
    const body = new FormData();
    body.append('question', question.trim());
    body.append('mode', 'full');
    if (inputType === 'url') body.append('url', url.trim());
    else body.append('file', file);
    try {
      setLoading(true);
      const response = await fetch(CASES_URL, {
        method: 'POST',
        headers: getResearchHeaders(API_KEY),
        body,
      });
      saveActiveCase(await readJson(response));
    } catch (submitError) {
      setError(submitError.message || '研究案件创建失败。');
    } finally {
      setLoading(false);
    }
  }

  function chooseType(type) {
    setInputType(type);
    setFile(null);
    if (fileRef.current) fileRef.current.value = '';
  }

  return (
    <section className="research-case-workspace" aria-label="统一研究案件入口">
      <div className="research-case-copy">
        <p className="eyebrow">Unified Research Case</p>
        <h2>创建统一研究案件</h2>
        <p>
          URL、PDF或图片先经过8层证据护盾，再进入宏观、资金流、长期基本面和量化模型专家；
          结果会在四个工作台共享同一个案件编号与数据时点。
        </p>
        <div className="research-loop-line" aria-label="研究闭环">
          <span>采集</span><i />
          <span>门控</span><i />
          <span>多Agent</span><i />
          <span>三期限</span><i />
          <span>个性化</span><i />
          <span>前向验证</span>
        </div>
      </div>
      <form className="research-case-form" onSubmit={submit} aria-busy={loading}>
        <div className="case-input-tabs" role="group" aria-label="证据输入类型">
          <button type="button" className={inputType === 'url' ? 'active' : ''} onClick={() => chooseType('url')}>
            <Globe2 size={15} /> URL
          </button>
          <button type="button" className={inputType === 'pdf' ? 'active' : ''} onClick={() => chooseType('pdf')}>
            <FileText size={15} /> PDF
          </button>
          <button type="button" className={inputType === 'image' ? 'active' : ''} onClick={() => chooseType('image')}>
            <FileImage size={15} /> 图片
          </button>
        </div>
        <label>
          <span>研究问题</span>
          <textarea name="question" autoComplete="off" aria-label="研究问题" rows="2" value={question} onChange={(event) => setQuestion(event.target.value)} />
        </label>
        {inputType === 'url' ? (
          <label>
            <span>证据 URL · 仅HTTPS公网</span>
            <input name="evidence_url" autoComplete="off" spellCheck="false" aria-label="证据 URL" type="url" placeholder="例如：https://www.federalreserve.gov/…" value={url} onChange={(event) => setUrl(event.target.value)} />
          </label>
        ) : (
          <label>
            <span>{inputType === 'pdf' ? 'PDF · 最大20MB' : 'PNG/JPEG/WebP · 最大10MB'}</span>
            <input
              ref={fileRef}
              aria-label={inputType === 'pdf' ? '上传 PDF' : '上传图片'}
              name="evidence_file"
              type="file"
              accept={inputType === 'pdf' ? 'application/pdf' : 'image/png,image/jpeg,image/webp'}
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
          </label>
        )}
        <button className="case-run-button" type="submit" disabled={loading}>
          {loading ? <Loader2 size={16} className="spinning" /> : <Sparkles size={16} />}
          {loading ? '证据门控与专家研究中' : '运行研究闭环'}
        </button>
        {error ? <p className="case-form-error" role="alert"><AlertTriangle size={14} />{error}</p> : null}
      </form>
      {activeCase ? (
        <div className="case-workspace-result" aria-live="polite">
          <ActiveCaseRibbon activeCase={activeCase} />
          {activeCase.narrative ? (
            <article className="case-narrative" aria-label="研究叙事摘要">
              <span>{activeCase.narrative.generated_by === 'llm' ? 'Full · LLM叙事已审计' : 'Draft · 确定性叙事'}</span>
              <p>{activeCase.narrative.overview}</p>
              {activeCase.narrative.degradation_flags?.length ? (
                <small>降级：{activeCase.narrative.degradation_flags.join(' · ')}</small>
              ) : null}
            </article>
          ) : null}
          <EvidenceShieldPanel activeCase={activeCase} />
        </div>
      ) : null}
    </section>
  );
}

export function ActiveCaseRibbon({ activeCase: suppliedCase = null }) {
  const storedCase = useActiveResearchCase();
  const activeCase = suppliedCase || storedCase;
  if (!activeCase) return null;
  const blocked = activeCase.status === 'blocked';
  return (
    <section className={`active-case-ribbon ${blocked ? 'blocked' : ''}`} aria-label="当前研究案件">
      <div className="case-ribbon-icon">{blocked ? <ShieldAlert size={18} /> : <Fingerprint size={18} />}</div>
      <div>
        <span>当前研究案件</span>
        <strong>{activeCase.case_id}</strong>
      </div>
      <div>
        <span>研究问题</span>
        <strong>{activeCase.question}</strong>
      </div>
      <div>
        <span>数据时点</span>
        <strong>{activeCase.data_asof || '未提供'}</strong>
      </div>
      <div className={`case-status status-${activeCase.status}`}>
        {statusLabels[activeCase.status] || activeCase.status}
      </div>
    </section>
  );
}

export function EvidenceShieldPanel({ activeCase: suppliedCase = null }) {
  const storedCase = useActiveResearchCase();
  const activeCase = suppliedCase || storedCase;
  const [open, setOpen] = useState(true);
  if (!activeCase) return null;
  const gates = activeCase.gate_report || [];
  const passed = gates.filter((gate) => gate.decision === 'pass').length;
  return (
    <section className="evidence-shield-panel">
      <button type="button" className="shield-panel-head" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <span><ShieldCheck size={17} /><h2>Evidence Shield · 8层证据护盾</h2></span>
        <small>{passed}/{gates.length || 8} 通过 · 点击{open ? '收起' : '展开'}</small>
      </button>
      {open ? (
        <div className="shield-gate-grid">
          {gates.map((gate) => (
            <article key={gate.gate} className={`shield-gate gate-${gate.decision}`} title={gate.reason}>
              {gate.decision === 'pass' ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
              <div>
                <strong>{gateLabels[gate.gate] || gate.gate}</strong>
                <span>{gate.decision} · 置信乘数 {Number(gate.confidence_multiplier).toFixed(2)}</span>
              </div>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  );
}

export function ModelGovernancePanel() {
  const activeCase = useActiveResearchCase();
  const registry = activeCase?.model_registry || [];
  return (
    <section className="panel-block model-registry-panel">
      <div className="panel-title">
        <Layers3 size={16} />
        <div><h2>模型冠军—挑战者治理</h2><span>样本外证据优先 · 深度模型不可越过硬门</span></div>
      </div>
      {registry.length ? (
        <div className="model-registry-grid">
          {registry.map((model) => (
            <article key={model.model_id} className={`model-card state-${model.state}`}>
              <span>{model.role === 'champion' ? '冠军模型' : '挑战者'}</span>
              <strong>{model.label}</strong>
              <em>{modelStateLabels[model.state] || model.state}</em>
              <small>{model.oos_metric || model.degradation_reason || '等待样本外评估'}</small>
              <p>{model.can_influence_strategy ? '可在门控后参与策略' : '仅观察，不改变策略'}</p>
            </article>
          ))}
        </div>
      ) : <p className="muted-copy">创建研究案件后显示本次使用的生产模型、挑战者与降级原因。</p>}
    </section>
  );
}

export function CaseStrategyPanel() {
  const activeCase = useActiveResearchCase();
  if (!activeCase) return null;
  const strategies = activeCase.horizon_strategy || {};
  return (
    <section className="case-strategy-panel">
      <div className="section-head">
        <div><span>Shared Case Strategy</span><h2>当前案件三期限策略</h2></div>
        {activeCase.conflicts?.length ? <strong className="minority-badge">少数意见保留</strong> : null}
      </div>
      <div className="case-strategy-grid">
        {Object.entries(strategies).map(([key, strategy]) => (
          <article key={key}>
            <span>{horizonLabels[key] || key}</span>
            <strong>{strategy.stance} · {Math.round((strategy.confidence || 0) * 100)}%</strong>
            <p>{strategy.base?.description}</p>
            <div className="scenario-weight-audit">
              <b>{strategy.base?.probability_kind === 'calibrated_probability' ? '校准概率' : strategy.base?.probability_kind === 'abstention' ? '弃权分配' : '研究权重 · 非校准概率'}</b>
              <span>基准 {Math.round((strategy.base?.probability || 0) * 100)} · 上行 {Math.round((strategy.upside?.probability || 0) * 100)} · 下行 {Math.round((strategy.downside?.probability || 0) * 100)}</span>
              <code>{strategy.base?.method || 'method unavailable'}</code>
            </div>
            <small>方向一致性：{strategy.priced_in} · 下次复核 {strategy.next_review_at?.slice(0, 10)}</small>
            <small>置信来源：{strategy.confidence_basis?.method || '未记录'} · {strategy.confidence_basis?.calibrated ? '已校准' : '未校准'}</small>
            <ul>{(strategy.invalidation || []).slice(0, 2).map((line) => <li key={line}>{line}</li>)}</ul>
          </article>
        ))}
      </div>
      <section className="agent-evidence-ledger">
        <h2>Agent 专属证据账本</h2>
        <div>
          {(activeCase.agent_views || []).map((view) => (
            <article key={`${view.agent}-${view.horizon}`}>
              <span>{agentLabels[view.agent] || view.agent}</span>
              <strong>{view.stance} · {Math.round((view.confidence || 0) * 100)}%</strong>
              <p>{view.thesis}</p>
              <small>专属支持 {view.supporting_fact_ids?.length || 0} · 反方 {view.counter_fact_ids?.length || 0}</small>
              <code>{view.confidence_basis?.method || '未记录置信来源'}</code>
            </article>
          ))}
        </div>
      </section>
      {activeCase.outcome_schedule?.length ? (
        <section className="case-outcome-audit">
          <h2>案件前向校准</h2>
          <div>{activeCase.outcome_schedule.map((checkpoint) => (
            <article key={`${checkpoint.horizon}-${checkpoint.due_at}`}>
              <span>{horizonLabels[checkpoint.horizon] || checkpoint.horizon}</span>
              <strong>{checkpoint.status === 'scored'
                ? checkpoint.scenario_score != null
                  ? `${checkpoint.scenario_score_kind === 'brier' ? '概率' : '权重'} Brier ${Number(checkpoint.scenario_score).toFixed(3)}`
                  : '已到期 · 评分不可用'
                : '等待到期'}</strong>
              <small>{checkpoint.status === 'scored' ? `实际 ${checkpoint.scenario_outcome} · 收益 ${(Number(checkpoint.realized_return) * 100).toFixed(2)}%` : `到期 ${checkpoint.due_at?.slice(0, 10)}`}</small>
            </article>
          ))}</div>
        </section>
      ) : null}
      {(activeCase.conflicts || []).map((conflict) => (
        <div className="case-conflict" key={conflict.topic}>
          <AlertTriangle size={15} /><div><strong>{conflict.topic}</strong><p>{conflict.minority_view}</p><small>{conflict.resolution}</small></div>
        </div>
      ))}
    </section>
  );
}

export function ThreeDimensionalBrief({ brief }) {
  const sections = useMemo(() => [
    { key: 'agent', title: 'Agent解释层', icon: BrainCircuit, body: brief?.agent?.core_conclusion, foot: '情景解释与重点排序' },
    { key: 'rules', title: '硬规则层', icon: ShieldCheck, body: (brief?.rules?.hard_constraints || []).join(' '), foot: `${brief?.rules?.risk_flags?.length || 0} 项风险旗标` },
    { key: 'api', title: '实时API层', icon: DatabaseZap, body: `数据截至 ${brief?.api?.data_asof || '未知'} · ${brief?.api?.freshness || '未知新鲜度'}`, foot: `${brief?.api?.model_states?.length || 0} 个模型状态已复核` },
  ], [brief]);
  if (!brief) return null;
  return (
    <section className="three-dimensional-brief">
      <div className="panel-title">
        <Radar size={16} /><div><h2>Agent × 规则 × API 三维建议</h2><span>同一研究案件 · 三层相互约束</span></div>
      </div>
      <div className="three-d-grid">
        {sections.map(({ key, title, icon: Icon, body, foot }) => (
          <article key={key}><Icon size={18} /><strong>{title}</strong><p>{body}</p><small>{foot}</small></article>
        ))}
      </div>
      <div className="brief-watchline">
        <strong>观察清单</strong><span>{(brief.watchlist || []).join(' · ') || '等待可信触发器'}</span>
        <strong>失效条件</strong><span>{(brief.invalidation || []).join(' · ') || '当前无可验证条件'}</span>
      </div>
    </section>
  );
}
