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
const lossCapacityOptions = [['', '未填写'], ['low', '低'], ['medium', '中'], ['high', '高']];
const portfolioContextOptions = [['', '未确认'], ['true', '已覆盖全部主要资产'], ['false', '仅提供黄金仓位']];
const liabilitiesOptions = [['', '未填写'], ['low', '低'], ['medium', '中'], ['high', '高']];
const instrumentOptions = [
  ['', '未填写'], ['physical', '实物金'], ['unlevered_etf', '无杠杆黄金ETF'],
  ['unallocated_spot', '无杠杆账户金'], ['futures', '黄金期货'], ['options', '黄金期权'],
  ['cfd', '黄金CFD'], ['other', '其他'],
];
const jurisdictionOptions = [['', '未填写'], ['CN', '中国大陆'], ['SG', '新加坡'], ['HK', '中国香港'], ['US', '美国'], ['OTHER', '其他']];
const currencyOptions = [['', '未填写'], ['CNY', '人民币 CNY'], ['USD', '美元 USD'], ['SGD', '新元 SGD'], ['HKD', '港币 HKD'], ['OTHER', '其他']];

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
        适当性画像（补全后才显示个人仓位差距）
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
          <SelectRow
            label="损失承受能力"
            value={profile.loss_capacity || ''}
            options={lossCapacityOptions}
            onChange={(v) => onChange('loss_capacity', v || null)}
          />
          <SelectRow
            label="组合上下文"
            value={profile.portfolio_context_known == null ? '' : String(profile.portfolio_context_known)}
            options={portfolioContextOptions}
            onChange={(v) => onChange('portfolio_context_known', v === '' ? null : v === 'true')}
          />
          <label className="field compact-field">
            <span>应急资金月数</span>
            <div className="number-input">
              <input
                type="number"
                min="0"
                max="120"
                step="1"
                placeholder="未填写"
                value={profile.emergency_fund_months ?? ''}
                onChange={(event) => onChange('emergency_fund_months', event.target.value === '' ? null : event.target.value)}
              />
              <small>月</small>
            </div>
          </label>
          <SelectRow label="负债水平" value={profile.liabilities_level || ''} options={liabilitiesOptions} onChange={(v) => onChange('liabilities_level', v || null)} />
          <SelectRow label="黄金工具" value={profile.gold_instrument || ''} options={instrumentOptions} onChange={(v) => onChange('gold_instrument', v || null)} />
          <SelectRow label="运营法域" value={profile.jurisdiction || ''} options={jurisdictionOptions} onChange={(v) => onChange('jurisdiction', v || null)} />
          <SelectRow label="基础货币" value={profile.base_currency || ''} options={currencyOptions} onChange={(v) => onChange('base_currency', v || null)} />
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
