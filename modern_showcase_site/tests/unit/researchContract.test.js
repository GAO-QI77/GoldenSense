import test from 'node:test';
import assert from 'node:assert/strict';

import {
  calibrationProgressCopy,
  calibrationPendingCopy,
  classifySourceHealth,
  dataQualityStatusCopy,
  degradationFlagCopy,
  deriveApiEndpoint,
  formatPublicationDate,
  isCalibrationReady,
  isPositionAnalysisAllowed,
  isSignalPublicationEmpty,
  isSignalPublication,
  positionGapCopy,
  sourceReasonCopy,
} from '../../src/researchContract.js';

test('API endpoints are derived from the configured agent origin', () => {
  assert.equal(
    deriveApiEndpoint('https://api.example.com/api/v1/agent/analyze', '/api/v1/signals'),
    'https://api.example.com/api/v1/signals',
  );
  assert.equal(
    deriveApiEndpoint('https://api.example.com/api/v1/agent/analyze', '/api/v1/subscriptions'),
    'https://api.example.com/api/v1/subscriptions',
  );
  assert.equal(
    deriveApiEndpoint('/api/v1/agent/analyze', '/api/v1/signals'),
    '/api/v1/signals',
  );
  assert.equal(
    deriveApiEndpoint('https://api.example.com/api/v1/agent/analyze', 'api/v1/subscriptions'),
    'https://api.example.com/api/v1/subscriptions',
  );
});

test('signal publications require an identifier and valid publication date', () => {
  assert.equal(isSignalPublication({ status: 'empty' }), false);
  assert.equal(
    isSignalPublication({
      publication_id: '2026-W36',
      published_at: '2026-09-03T01:00:00Z',
    }),
    true,
  );
});

test('only the approved first-publication contract is a legitimate empty ledger', () => {
  assert.equal(
    isSignalPublicationEmpty({
      status: 'empty',
      error_code: 'first_publication_required',
    }),
    true,
  );
  assert.equal(isSignalPublicationEmpty({ status: 'empty' }), false);
  assert.equal(
    isSignalPublicationEmpty({ status: 'empty', error_code: 'route_not_found' }),
    false,
  );
});

test('publication dates omit missing and invalid values', () => {
  assert.equal(formatPublicationDate(undefined), null);
  assert.equal(formatPublicationDate('not-a-date'), null);
  assert.equal(formatPublicationDate('2026-09-03T01:00:00Z').includes('2026'), true);
});

test('position gap statuses have explicit public copy', () => {
  assert.equal(positionGapCopy({ status: 'within' }).label, '处于参考区间内');
  assert.equal(positionGapCopy({ status: 'above' }).label, '高于参考区间上沿');
  assert.equal(positionGapCopy({ status: 'below' }).label, '低于参考区间下沿');
  assert.equal(positionGapCopy({ status: 'withheld' }).label, '个人仓位差距暂不计算');
  assert.equal(positionGapCopy({ status: 'unknown' }).label, '个人仓位差距状态未知');
  assert.equal(positionGapCopy({ status: 'future_status' }).label, '个人仓位差距状态未知');
  assert.equal(positionGapCopy().label, '个人仓位差距状态未知');
});

test('source degradation reasons are translated with a safe fallback', () => {
  assert.equal(sourceReasonCopy('proxy_static_source'), '当前使用代理数据，并非实时官方数据');
  assert.equal(
    sourceReasonCopy('real_yield_proxy_from_nominal_yield'),
    '实际利率由名义利率估算，属于衍生代理值',
  );
  assert.equal(sourceReasonCopy('unexpected_reason'), '来源已降级，具体原因待核验');
  assert.equal(
    sourceReasonCopy({ status: 'degraded', degraded_reason: null }),
    '来源状态已降级，具体原因待核验',
  );
  assert.equal(
    sourceReasonCopy({ status: 'unavailable', degraded_reason: null }),
    '来源暂不可用',
  );
  assert.equal(sourceReasonCopy({ status: 'ok', degraded_reason: null }), '来源状态正常');
  assert.equal(sourceReasonCopy(), '来源状态待核验');
});

test('dashboard quality status and flags never expose machine codes', () => {
  assert.equal(dataQualityStatusCopy('ok'), '正常');
  assert.equal(dataQualityStatusCopy('degraded'), '部分降级');
  assert.equal(dataQualityStatusCopy('future_status'), '待核验');
  assert.equal(
    degradationFlagCopy('market_indicators_degraded'),
    '部分市场指标使用代理或降级数据',
  );
  assert.equal(degradationFlagCopy('future_machine_code'), '部分数据已降级，详情待核验');
});

test('source timeliness is independent from degraded provenance', () => {
  assert.deepEqual(
    classifySourceHealth({
      status: 'ok',
      freshness_seconds: 180,
      expected_lag_seconds: 180,
      degraded_reason: null,
    }),
    { timeliness: '按时', provenance: '直接来源' },
  );
  assert.deepEqual(
    classifySourceHealth({
      status: 'degraded',
      freshness_seconds: 11,
      expected_lag_seconds: 180,
      degraded_reason: 'real_yield_proxy_from_nominal_yield',
    }),
    { timeliness: '按时', provenance: '衍生值' },
  );
  assert.deepEqual(
    classifySourceHealth({
      status: 'degraded',
      freshness_seconds: '181',
      expected_lag_seconds: '180',
      degraded_reason: 'proxy_static_source',
    }),
    { timeliness: '延迟', provenance: '代理值' },
  );
  for (const invalidSource of [
    { freshness_seconds: -1, expected_lag_seconds: 180 },
    { freshness_seconds: '-1', expected_lag_seconds: '180' },
    { freshness_seconds: 'not-a-number', expected_lag_seconds: '180' },
    { freshness_seconds: '', expected_lag_seconds: '180' },
    { freshness_seconds: '11', expected_lag_seconds: '0' },
  ]) {
    assert.equal(classifySourceHealth(invalidSource).timeliness, '时效未知');
  }
  assert.deepEqual(classifySourceHealth({ status: 'unknown' }), {
    timeliness: '时效未知',
    provenance: '来源待核验',
  });
  assert.deepEqual(
    classifySourceHealth({
      status: 'unavailable',
      freshness_seconds: 200,
      expected_lag_seconds: 180,
    }),
    { timeliness: '延迟', provenance: '来源不可用' },
  );
  assert.deepEqual(classifySourceHealth(), {
    timeliness: '时效未知',
    provenance: '来源待核验',
  });
});

test('zero calibration samples are presented as normal accumulation', () => {
  assert.equal(
    calibrationProgressCopy({
      total_scored: 0,
      governance: { directional_calls: 0, min_directional_calls: 20 },
    }),
    '正常积累中 0/20',
  );
});

test('pending calibration analyses are explained without exposing unsafe values', () => {
  assert.equal(calibrationPendingCopy({ pending_analyses: 3 }), '另有 3 条分析等待复核日成熟。');
  assert.equal(calibrationPendingCopy({ pending_analyses: '2' }), '另有 2 条分析等待复核日成熟。');
  for (const pending of [0, -1, '', 'invalid', Infinity, undefined]) {
    assert.equal(calibrationPendingCopy({ pending_analyses: pending }), null);
  }
});

test('calibration readiness uses directional calls and a configurable minimum', () => {
  assert.equal(
    isCalibrationReady({ total_scored: 30, directional_calls: 19, minimum_directional_calls: 20 }),
    false,
  );
  assert.equal(
    isCalibrationReady({ total_scored: 30, directional_calls: '20', minimum_directional_calls: '20' }),
    true,
  );
  assert.equal(isCalibrationReady({ total_scored: 30, directional_calls: 0 }), false);
  assert.equal(isCalibrationReady({ total_scored: 30 }), false);
});

test('calibration progress parses safe numeric strings and rejects invalid counts', () => {
  assert.equal(
    calibrationProgressCopy({ directional_calls: ' 7 ', minimum_directional_calls: '20' }),
    '正常积累中 7/20',
  );
  assert.equal(
    calibrationProgressCopy({
      governance: { directional_calls: '12', min_directional_calls: '12' },
    }),
    '正常积累中 12/12',
  );
  for (const directionalCalls of [-1, '-1', '', 'invalid', Infinity]) {
    assert.equal(
      calibrationProgressCopy({ directional_calls: directionalCalls, minimum_directional_calls: 20 }),
      '正常积累中 0/20',
    );
  }
  for (const minimumDirectionalCalls of [-1, '0', '', 'invalid', Infinity]) {
    assert.equal(
      calibrationProgressCopy({ directional_calls: '7', minimum_directional_calls: minimumDirectionalCalls }),
      '正常积累中 7/20',
    );
  }
});

test('personal position analysis requires explicit suitability and a calculable status', () => {
  for (const status of ['within', 'above', 'below']) {
    assert.equal(
      isPositionAnalysisAllowed({ position_analysis_allowed: true }, { status }),
      true,
    );
  }
  assert.equal(
    isPositionAnalysisAllowed({ position_analysis_allowed: false }, { status: 'within' }),
    false,
  );
  for (const status of ['withheld', 'unknown', 'future_status', undefined]) {
    assert.equal(
      isPositionAnalysisAllowed({ position_analysis_allowed: true }, { status }),
      false,
    );
  }
  assert.equal(isPositionAnalysisAllowed(undefined, { status: 'within' }), false);
  assert.equal(isPositionAnalysisAllowed({ position_analysis_allowed: true }, undefined), false);
});
