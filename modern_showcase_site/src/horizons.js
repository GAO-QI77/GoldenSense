export const RESEARCH_HORIZON_LABELS = {
  short_term: '短期',
  mid_term: '中期',
  long_term: '长期',
};

export const RESEARCH_HORIZON_WINDOWS = {
  short_term: '1–21 天',
  mid_term: '1–6 月',
  long_term: '6 月以上',
};

export function horizonLabel(value) {
  return RESEARCH_HORIZON_LABELS[value] || '研究周期';
}

const FORECAST_WINDOW_LABELS = {
  '24h': '24 小时模型',
  '7d': '7 天模型',
  '30d': '30 天模型',
};

export function forecastWindowLabel(value) {
  return RESEARCH_HORIZON_LABELS[value] || FORECAST_WINDOW_LABELS[value] || '模型窗口';
}
