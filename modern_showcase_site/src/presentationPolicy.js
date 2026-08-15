const STATE_COPY = {
  loading: { label: '正在连接市场数据', tone: 'neutral' },
  current: { label: '当前可用', tone: 'positive' },
  delayed: { label: '数据已延迟', tone: 'warning' },
  degraded: { label: '部分数据降级', tone: 'warning' },
  unavailable: { label: '暂时无法获取当前市场', tone: 'danger' },
};

export function publicErrorMessage(error) {
  const technicalDetail = typeof error === 'string'
    ? error
    : error?.message || String(error || 'unknown error');
  return {
    summary: '暂时无法获取当前市场数据，请稍后重试。',
    technicalDetail,
  };
}

export function deriveDashboardState({ loading = false, error = null, dashboard = null } = {}) {
  let kind = 'current';
  if (error) kind = 'unavailable';
  else if (loading) kind = 'loading';
  else if (!dashboard) kind = 'unavailable';
  else if (dashboard.market_status?.status === 'unavailable') kind = 'unavailable';
  else if (dashboard.market_status?.is_stale) kind = 'delayed';
  else if (dashboard.data_quality?.status === 'degraded' || dashboard.degradation_flags?.length) kind = 'degraded';

  const message = kind === 'unavailable' ? publicErrorMessage(error) : null;
  return { kind, ...STATE_COPY[kind], message };
}

export function presentHalfLife(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0 || numeric > 3650) {
    return { available: false, label: '未观察到有效均值回归' };
  }
  const rounded = Math.round(numeric);
  return { available: true, label: `${rounded} 日`, value: rounded };
}

function outcomeKey(row) {
  if (row?.analysis_id) return `analysis:${row.analysis_id}`;
  return ['composite', row?.case_id, row?.horizon, row?.evaluated_at, row?.outcome]
    .map((part) => part ?? '')
    .join(':');
}

export function dedupeOutcomes(rows = []) {
  const seen = new Set();
  return rows.filter((row) => {
    const key = outcomeKey(row);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
