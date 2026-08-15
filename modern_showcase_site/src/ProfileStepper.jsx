import React from 'react';
import { CheckCircle2, ChevronLeft, ChevronRight, Circle } from 'lucide-react';

const STEP_META = [
  { number: 1, title: '了解你', hint: '期限、经验与当前黄金占比' },
  { number: 2, title: '补全边界', hint: '回撤、流动性、工具与法域' },
  { number: 3, title: '生成研究', hint: '选择适合的研究能力' },
];

export default function ProfileStepper({ step, onStepChange, completion, core, suitability, research }) {
  const completeByStep = [completion.coreComplete, completion.suitabilityComplete, false];
  const panels = [core, suitability, research];
  return (
    <section className="profile-stepper" aria-label="个性化研究三步流程">
      <div className="profile-progress-head">
        <div>
          <span>第 {step} 步，共 3 步</span>
          <strong>{STEP_META[step - 1].title}</strong>
        </div>
        <span>{completion.completedCount}/{completion.totalCount} 项已完成</span>
      </div>
      <div className="profile-step-tabs" role="tablist" aria-label="画像步骤">
        {STEP_META.map((item) => (
          <button
            key={item.number}
            type="button"
            role="tab"
            aria-selected={step === item.number}
            className={step === item.number ? 'active' : ''}
            onClick={() => onStepChange(item.number)}
          >
            {completeByStep[item.number - 1] ? <CheckCircle2 size={16} /> : <Circle size={16} />}
            <span><strong>{item.number}. {item.title}</strong><small>{item.hint}</small></span>
          </button>
        ))}
      </div>
      <div className="profile-step-panels">
        {panels.map((panel, index) => (
          <div
            key={STEP_META[index].number}
            className={`profile-step-panel ${step === index + 1 ? 'active' : ''}`}
            role="tabpanel"
            aria-label={`第 ${index + 1} 步：${STEP_META[index].title}`}
          >
            {panel}
          </div>
        ))}
      </div>
      <div className="profile-mobile-actions">
        <button type="button" onClick={() => onStepChange(Math.max(1, step - 1))} disabled={step === 1}>
          <ChevronLeft size={16} /> 上一步
        </button>
        <button type="button" onClick={() => onStepChange(Math.min(3, step + 1))} disabled={step === 3}>
          {step === 2 && !completion.suitabilityComplete ? '仍有信息未填' : '下一步'} <ChevronRight size={16} />
        </button>
      </div>
    </section>
  );
}
