import React, { useState } from 'react';
import { ChevronRight } from 'lucide-react';

export const riskOptions = [
  ['conservative', '保守型'],
  ['balanced', '稳健型'],
  ['aggressive', '进取型'],
];
export const horizonOptions = [
  ['short', '短期(1-21天)'],
  ['mid', '中期(1-6月)'],
  ['long', '长期(6月+)'],
];
export const experienceOptions = [
  ['novice', '新手'],
  ['experienced', '有经验'],
  ['professional', '专业'],
];
const liquidityOptions = [
  ['', '未填写'],
  ['low', '低'],
  ['medium', '中'],
  ['high', '高'],
];
const leverageOptions = [
  ['', '未填写'],
  ['none', '不用杠杆'],
  ['low', '低杠杆'],
  ['medium', '中等杠杆'],
  ['high', '高杠杆'],
];
const goalOptions = [
  ['', '未填写'],
  ['capital_preservation', '本金保护'],
  ['income', '稳健增值'],
  ['event_trade', '事件交易'],
  ['trend_following', '趋势跟随'],
  ['speculation', '投机博弈'],
];

export function SegmentedRow({ label, value, options, onChange }) {
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

export function CoreProfileFields({ profile, onChange }) {
  return (
    <>
      <SegmentedRow
        label="风险承受能力"
        value={profile.risk_tolerance}
        options={riskOptions}
        onChange={(v) => onChange('risk_tolerance', v)}
      />
      <SegmentedRow
        label="投资期限"
        value={profile.horizon}
        options={horizonOptions}
        onChange={(v) => onChange('horizon', v)}
      />
      <SegmentedRow
        label="投资经验"
        value={profile.experience}
        options={experienceOptions}
        onChange={(v) => onChange('experience', v)}
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
            onChange={(event) => onChange('current_gold_pct', event.target.value)}
          />
          <small>%</small>
        </div>
      </label>
    </>
  );
}

export function AdvancedProfileFields({ profile, onChange, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="advanced-profile-block">
      <button
        type="button"
        className="advanced-toggle"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <ChevronRight size={14} className={open ? 'open' : ''} />
        进阶画像（可选 · 填写后解锁更多风险规则）
      </button>
      {open ? (
        <div className="advanced-fields">
          <label className="field compact-field">
            <span>最大回撤承受力（组合 %）</span>
            <div className="number-input">
              <input
                type="number"
                min="0"
                max="100"
                step="0.5"
                placeholder="未填写"
                value={profile.max_drawdown_pct ?? ''}
                onChange={(event) =>
                  onChange('max_drawdown_pct', event.target.value === '' ? null : event.target.value)
                }
              />
              <small>%</small>
            </div>
          </label>
          <SelectRow
            label="流动性需求"
            value={profile.liquidity_need || ''}
            options={liquidityOptions}
            onChange={(v) => onChange('liquidity_need', v || null)}
          />
          <SelectRow
            label="杠杆态度"
            value={profile.leverage_attitude || ''}
            options={leverageOptions}
            onChange={(v) => onChange('leverage_attitude', v || null)}
          />
          <SelectRow
            label="投资目标"
            value={profile.investment_goal || ''}
            options={goalOptions}
            onChange={(v) => onChange('investment_goal', v || null)}
          />
        </div>
      ) : null}
    </div>
  );
}

function SelectRow({ label, value, options, onChange }) {
  return (
    <label className="field compact-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map(([optionValue, optionLabel]) => (
          <option key={optionValue} value={optionValue}>
            {optionLabel}
          </option>
        ))}
      </select>
    </label>
  );
}
