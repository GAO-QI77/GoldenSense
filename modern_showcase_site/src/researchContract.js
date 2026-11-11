export function deriveApiEndpoint(analyzeUrl, endpointPath) {
  const path = endpointPath.startsWith('/') ? endpointPath : `/${endpointPath}`;
  try {
    return new URL(path, new URL(analyzeUrl).origin).toString();
  } catch {
    return path;
  }
}

export function isSignalPublication(value) {
  return Boolean(value?.publication_id && formatPublicationDate(value?.published_at));
}

export function isSignalPublicationEmpty(value) {
  return value?.status === 'empty' && value?.error_code === 'first_publication_required';
}

export function formatPublicationDate(value) {
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) return null;
  return date.toLocaleString('zh-CN', { hour12: false });
}

const POSITION_GAP_COPY = Object.freeze({
  within: { label: '处于参考区间内', tone: 'bull' },
  above: { label: '高于参考区间上沿', tone: 'risk' },
  below: { label: '低于参考区间下沿', tone: 'neutral' },
  withheld: { label: '个人仓位差距暂不计算', tone: 'neutral' },
  unknown: { label: '个人仓位差距状态未知', tone: 'neutral' },
});

const CALCULABLE_POSITION_STATUSES = new Set(['within', 'above', 'below']);
const DEFAULT_MINIMUM_DIRECTIONAL_CALLS = 20;

const SOURCE_REASON_COPY = Object.freeze({
  proxy_static_source: '当前使用代理数据，并非实时官方数据',
  real_yield_proxy_from_nominal_yield: '实际利率由名义利率估算，属于衍生代理值',
  derived_support_resistance_proxy: '支撑与阻力位由行情推导，属于衍生值',
  market_snapshot_stale: '市场快照已超过预期更新时限',
  market_snapshot_synthetic_fallback: '市场快照使用合成回退数据',
  synthetic_fallback: '当前使用合成回退数据',
  recent_news_unavailable: '近期新闻来源暂不可用',
  market_snapshot_unavailable: '市场快照来源暂不可用',
  upstream_unavailable: '上游来源暂不可用',
});

const SOURCE_PROVENANCE = Object.freeze({
  proxy_static_source: '代理值',
  real_yield_proxy_from_nominal_yield: '衍生值',
  derived_support_resistance_proxy: '衍生值',
  market_snapshot_stale: '直接来源',
  market_snapshot_synthetic_fallback: '代理值',
  synthetic_fallback: '代理值',
});

const DATA_QUALITY_STATUS_COPY = Object.freeze({
  ok: '正常',
  degraded: '部分降级',
  unavailable: '不可用',
  error: '异常',
  unknown: '待核验',
});

const DEGRADATION_FLAG_COPY = Object.freeze({
  market_indicators_degraded: '部分市场指标使用代理或降级数据',
  gold_history_degraded: '黄金历史行情暂时降级',
  quant_forecast_degraded: '量化预测暂时使用降级结果',
  sources_degraded: '部分数据来源暂时降级',
  production_model_degraded: '生产模型暂时使用保守降级结果',
});

export function positionGapCopy(positionGap) {
  return POSITION_GAP_COPY[positionGap?.status] || POSITION_GAP_COPY.unknown;
}

export function isPositionAnalysisAllowed(suitability, positionGap) {
  return (
    suitability?.position_analysis_allowed === true &&
    CALCULABLE_POSITION_STATUSES.has(positionGap?.status)
  );
}

export function sourceReasonCopy(sourceOrReason) {
  const source =
    sourceOrReason && typeof sourceOrReason === 'object' ? sourceOrReason : undefined;
  const reason = source ? source.degraded_reason : sourceOrReason;
  if (source?.status === 'unavailable') return '来源暂不可用';
  if (SOURCE_REASON_COPY[reason]) return SOURCE_REASON_COPY[reason];
  if (typeof reason === 'string' && reason.includes('synthetic_fallback')) {
    return SOURCE_REASON_COPY.synthetic_fallback;
  }
  if (source?.status === 'degraded') return '来源状态已降级，具体原因待核验';
  if (source?.status === 'ok') return '来源状态正常';
  if (reason || (!source && typeof sourceOrReason === 'string')) {
    return '来源已降级，具体原因待核验';
  }
  return '来源状态待核验';
}

export function dataQualityStatusCopy(status) {
  return DATA_QUALITY_STATUS_COPY[status] || DATA_QUALITY_STATUS_COPY.unknown;
}

export function degradationFlagCopy(flag) {
  return DEGRADATION_FLAG_COPY[flag] || '部分数据已降级，详情待核验';
}

function parseSafeNonNegativeNumber(value) {
  if (typeof value === 'string') {
    const normalized = value.trim();
    if (!/^\d+(?:\.\d+)?$/.test(normalized)) return null;
    value = Number(normalized);
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return null;
  return value;
}

function parseSafeCount(value) {
  const parsed = parseSafeNonNegativeNumber(value);
  return Number.isInteger(parsed) ? parsed : null;
}

function firstValidCount(values, { positive = false } = {}) {
  for (const value of values) {
    const parsed = parseSafeCount(value);
    if (parsed !== null && (!positive || parsed > 0)) return parsed;
  }
  return null;
}

function calibrationCounts(calibration) {
  const current = firstValidCount([
    calibration?.governance?.directional_calls,
    calibration?.directional_calls,
  ]);
  const target = firstValidCount(
    [
      calibration?.governance?.min_directional_calls,
      calibration?.minimum_directional_calls,
      calibration?.min_directional_calls,
    ],
    { positive: true },
  );
  return {
    current: current ?? 0,
    target: target ?? DEFAULT_MINIMUM_DIRECTIONAL_CALLS,
  };
}

export function isCalibrationReady(calibration) {
  if (!calibration) return false;
  const { current, target } = calibrationCounts(calibration);
  return current >= target;
}

export function calibrationProgressCopy(calibration) {
  const { current, target } = calibrationCounts(calibration);
  return `正常积累中 ${current}/${target}`;
}

export function calibrationPendingCopy(calibration) {
  const pending = parseSafeCount(calibration?.pending_analyses);
  return pending && pending > 0 ? `另有 ${pending} 条分析等待复核日成熟。` : null;
}

export function sourceTimelinessCopy(source) {
  const freshness = parseSafeNonNegativeNumber(source?.freshness_seconds);
  const expectedLag = parseSafeNonNegativeNumber(source?.expected_lag_seconds);
  if (freshness === null || expectedLag === null || expectedLag <= 0) {
    return '时效未知';
  }
  return freshness <= expectedLag ? '按时' : '延迟';
}

export function sourceProvenanceCopy(source) {
  const reason = source?.degraded_reason;
  if (SOURCE_PROVENANCE[reason]) return SOURCE_PROVENANCE[reason];
  if (typeof reason === 'string' && (reason.includes('synthetic_fallback') || reason.includes('proxy'))) {
    return '代理值';
  }
  if (source?.status === 'unavailable') return '来源不可用';
  if (source?.status === 'ok' && !reason) return '直接来源';
  return '来源待核验';
}

export function classifySourceHealth(source) {
  return {
    timeliness: sourceTimelinessCopy(source),
    provenance: sourceProvenanceCopy(source),
  };
}
