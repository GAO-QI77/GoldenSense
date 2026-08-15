import test from 'node:test';
import assert from 'node:assert/strict';

import {
  dedupeOutcomes,
  deriveDashboardState,
  presentHalfLife,
  publicErrorMessage,
} from '../../src/presentationPolicy.js';
import { defaultProfile, getProfileCompletion } from '../../src/profileStore.js';

test('dashboard error cannot be presented as current', () => {
  assert.equal(
    deriveDashboardState({ loading: false, error: 'boom', dashboard: { data_quality: { status: 'ok' } } }).kind,
    'unavailable',
  );
});

test('dashboard state distinguishes current, delayed and degraded data', () => {
  assert.equal(deriveDashboardState({ loading: true }).kind, 'loading');
  assert.equal(deriveDashboardState({ dashboard: { market_status: { is_stale: true } } }).kind, 'delayed');
  assert.equal(deriveDashboardState({ dashboard: { data_quality: { status: 'degraded' } } }).kind, 'degraded');
  assert.equal(deriveDashboardState({ dashboard: { data_quality: { status: 'ok' } } }).kind, 'current');
});

test('internal transport errors become recoverable public copy', () => {
  const message = publicErrorMessage('RuntimeError: get_market_snapshot:ConnectError');
  assert.doesNotMatch(message.summary, /RuntimeError|ConnectError/);
  assert.match(message.summary, /暂时无法/);
  assert.match(message.technicalDetail, /ConnectError/);
});

test('implausible half life abstains', () => {
  assert.deepEqual(presentHalfLife(53772), {
    available: false,
    label: '未观察到有效均值回归',
  });
});

test('plausible half life remains available', () => {
  assert.deepEqual(presentHalfLife(42), { available: true, label: '42 日', value: 42 });
});

test('duplicate calibration outcomes are shown once', () => {
  const rows = [
    { analysis_id: 'a', outcome: 'win' },
    { analysis_id: 'a', outcome: 'win' },
    { case_id: 'b', horizon: 'mid_term', evaluated_at: '2026-08-15' },
    { case_id: 'b', horizon: 'mid_term', evaluated_at: '2026-08-15' },
  ];
  assert.equal(dedupeOutcomes(rows).length, 2);
});

test('default profile is explicitly incomplete until suitability fields are supplied', () => {
  const incomplete = getProfileCompletion(defaultProfile);
  assert.equal(incomplete.coreComplete, true);
  assert.equal(incomplete.suitabilityComplete, false);
  assert.ok(incomplete.suitabilityMissing.includes('max_drawdown_pct'));

  const completed = getProfileCompletion({
    ...defaultProfile,
    max_drawdown_pct: 12,
    liquidity_need: 'medium',
    leverage_attitude: 'none',
    investment_goal: 'capital_preservation',
    loss_capacity: 'medium',
    portfolio_context_known: true,
    emergency_fund_months: 9,
    liabilities_level: 'low',
    gold_instrument: 'unlevered_etf',
    jurisdiction: 'SG',
    base_currency: 'SGD',
  });
  assert.equal(completed.suitabilityComplete, true);
  assert.deepEqual(completed.suitabilityMissing, []);
});
