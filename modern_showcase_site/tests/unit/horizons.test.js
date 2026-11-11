import assert from 'node:assert/strict';
import test from 'node:test';

import { forecastWindowLabel, horizonLabel } from '../../src/horizons.js';

test('keeps research categories independent of model windows', () => {
  assert.equal(horizonLabel('short_term'), '短期');
  assert.equal(horizonLabel('mid_term'), '中期');
  assert.equal(horizonLabel('long_term'), '长期');
  assert.equal(horizonLabel('30d'), '研究周期');
  assert.equal(forecastWindowLabel('24h'), '24 小时模型');
  assert.equal(forecastWindowLabel('7d'), '7 天模型');
  assert.equal(forecastWindowLabel('30d'), '30 天模型');
});

test('falls back safely when an unknown horizon is returned', () => {
  assert.equal(horizonLabel('experimental'), '研究周期');
  assert.equal(horizonLabel(null), '研究周期');
});
