import { expect, test } from '@playwright/test';

const dashboardPayload = {
  as_of: '2026-04-30T08:00:00Z',
  market_status: {
    asset: 'XAUUSD',
    as_of: '2026-04-30T08:00:00Z',
    latest_price: 2368.42,
    price_change_pct_1d: 0.004,
    freshness_seconds: 18,
    is_stale: false,
    status: 'ok',
    degraded_reason: null,
  },
  horizon_forecasts: ['24h', '7d', '30d'].map((horizon, index) => ({
    horizon,
    stance: index === 2 ? '中性' : '偏多',
    confidence_band: index === 0 ? '高' : '中',
    action: index === 2 ? '观望' : '小仓试探',
    probability: index === 0 ? 0.68 : 0.61,
    basis: 'heuristic_proxy',
    model_status: horizon === '30d' ? 'not_applicable' : 'heuristic_proxy',
    model_loaded: false,
    model_checkpoint_path: 'model_checkpoints',
    reasons: [
      `${horizon} 当前使用代理预测，主要参考趋势、美元、利率和自动抓取的新闻环境。`,
      '市场快照未受用户问题改写。',
      '该预测用于研究基线。',
    ],
  })),
  indicator_groups: [
    ['fundamental', '基本面'],
    ['technical', '技术面'],
    ['macro_policy', '宏观政策'],
    ['flow_sentiment', '资金情绪'],
  ].map(([id, title]) => ({
    id,
    title,
    summary: `${title} mock summary`,
    score: 0.15,
    status: id === 'fundamental' ? 'degraded' : 'ok',
    freshness_seconds: 18,
    degraded_reason: id === 'fundamental' ? 'proxy_static_source' : null,
    indicators: ['A', 'B', 'C', 'D'].map((suffix) => ({
      id: `${id}-${suffix}`,
      label: `${title}${suffix}`,
      value: suffix === 'A' ? 'proxy' : 'ok',
      numeric_value: 0.1,
      unit: null,
      direction: suffix === 'D' ? 'risk' : 'neutral',
      source: 'playwright-mock',
      source_url: null,
      freshness_seconds: 18,
      status: suffix === 'A' ? 'degraded' : 'ok',
      degraded_reason: suffix === 'A' ? 'mock proxy' : null,
    })),
  })),
  recent_news: [
    {
      event_id: 'n1',
      published_at: '2026-04-30T07:30:00Z',
      title: 'Fed officials discuss real yields',
      summary: 'Mock macro news for gold.',
      source: 'mock-wire',
      normalized_event: 'real yields',
      sentiment_score: 0.2,
      importance: 0.8,
      categories: ['macro'],
      url: null,
    },
  ],
  citations: [
    {
      id: 'ind-wgc',
      label: 'World Gold Council proxy',
      source_type: 'market_indicators',
      excerpt: 'Mock WGC citation.',
      url: 'https://www.gold.org/',
    },
  ],
  source_health: [
    {
      id: 'market_snapshot',
      label: 'Market Snapshot',
      source_type: 'market_snapshot',
      status: 'ok',
      freshness_seconds: 18,
      expected_lag_seconds: 180,
      cadence: '日内',
      degraded_reason: null,
      coverage: ['XAUUSD', 'DXY', 'VIX'],
      url: null,
    },
    {
      id: 'wgc_gold_demand',
      label: 'WGC Gold Demand',
      source_type: 'fundamental',
      status: 'degraded',
      freshness_seconds: 86400,
      expected_lag_seconds: 2678400,
      cadence: '月度/季度',
      degraded_reason: 'proxy_static_source',
      coverage: ['央行购金', 'ETF flows'],
      url: 'https://www.gold.org/',
    },
    {
      id: 'cftc_cot',
      label: 'CFTC COT',
      source_type: 'flow_sentiment',
      status: 'degraded',
      freshness_seconds: 86400,
      expected_lag_seconds: 604800,
      cadence: '周度',
      degraded_reason: 'proxy_static_source',
      coverage: ['Managed Money'],
      url: 'https://www.cftc.gov/',
    },
  ],
  gold_history: {
    asset: 'XAUUSD',
    as_of: '2026-04-30T08:00:00Z',
    source: 'yfinance',
    points: [
      { date: '2026-04-24', price: 2300, change_pct: null },
      { date: '2026-04-25', price: 2310, change_pct: 0.0043 },
      { date: '2026-04-26', price: 2368, change_pct: 0.0251 },
      { date: '2026-04-27', price: 2378, change_pct: 0.0042 },
      { date: '2026-04-28', price: 2324, change_pct: -0.0227 },
      { date: '2026-04-29', price: 2382, change_pct: 0.025 },
    ],
    key_nodes: [
      {
        date: '2026-04-26',
        price: 2368,
        change_pct: 0.0251,
        direction: 'up',
        reason: '黄金单日上涨 +2.51%，主要因素：美元走弱支撑黄金。',
        factors: ['美元走弱支撑黄金', '美债收益率回落'],
      },
      {
        date: '2026-04-28',
        price: 2324,
        change_pct: -0.0227,
        direction: 'down',
        reason: '黄金单日下跌 -2.27%，主要因素：美元走强压制黄金。',
        factors: ['美元走强压制黄金'],
      },
    ],
  },
  data_quality: {
    status: 'degraded',
    degraded_tools: ['get_market_indicators'],
    freshness_seconds: 18,
    indicator_status: 'degraded',
    news_status: 'ok',
  },
  degradation_flags: ['market_indicators_degraded'],
  timing_ms: { total: 42 },
};

const analysisPayload = {
  analysis_id: 'analysis-playwright',
  summary_card: {
    stance: '高风险观望',
    horizon: '24h',
    confidence_band: '低',
    action: '观望',
    reasons: [
      '完整问卷触发风险画像门控：资金占比过高。',
      '24h 当前采用代理预测，主要参考趋势、美元、利率和自动抓取的新闻环境。',
    ],
    invalidators: [
      '如果美元和实际利率同步快速走强，当前观点需要重新评估。',
      '如果 VIX 升到 30 以上或数据明显陈旧，应立即降级为观望。',
    ],
    disclaimer: '本内容仅用于帮助理解黄金市场，不构成个性化投资建议。',
  },
  horizon_forecasts: dashboardPayload.horizon_forecasts,
  recent_news: dashboardPayload.recent_news,
  evidence_cards: [
    {
      id: 'ev-risk',
      title: '用户风险画像',
      signal_type: 'risk',
      takeaway: '问卷门控等级 high。',
      direction: 'neutral',
      citation_ids: ['cit-risk'],
    },
    {
      id: 'ev-quant',
      title: '量化预测',
      signal_type: 'quant',
      takeaway: '代理量化引擎使用真实行情驱动，但问卷限制执行强度。',
      direction: 'supportive',
      citation_ids: ['cit-quant'],
    },
  ],
  citations: [
    {
      id: 'cit-risk',
      label: '用户风险画像',
      source_type: 'risk_profile',
      excerpt: '资金占比过高，最大回撤很低。',
      url: null,
    },
  ],
  risk_banner: {
    level: 'high',
    title: '问卷风险门控',
    message: '完整风险问卷显示本轮暴露与承受能力不匹配。',
  },
  degradation_flags: [],
  follow_up_questions: ['如果已经持仓，如何调整？'],
  timing_ms: { total: 90 },
};

test.beforeEach(async ({ page }) => {
  await page.route('**/api/v1/agent/dashboard/current', async (route) => {
    await route.fulfill({ json: dashboardPayload });
  });
  await page.route('**/api/v1/agent/analyze', async (route) => {
    const body = route.request().postDataJSON();
    expect(body.investor_profile).toBeTruthy();
    expect(body.investor_profile.capital_allocation_pct).toBeGreaterThanOrEqual(0);
    await route.fulfill({ json: analysisPayload });
  });
  await page.route('**/api/v1/agent/feedback', async (route) => {
    await route.fulfill({ json: { analysis_id: 'analysis-playwright', status: 'recorded' } });
  });
});

test('dashboard presents forecasts and four indicator pillars', async ({ page }) => {
  await page.goto('/');

  await expect(page.getByRole('heading', { name: '黄金价格预测与指标总览' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '今日核心结论' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '驱动贡献矩阵' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '风险催化日历' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '三情景推演' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '黄金价格走势' })).toBeVisible();
  await expect(page.locator('.key-node-panel').getByText('关键波动节点')).toBeVisible();
  await expect(page.locator('.key-node-panel').getByText('美元走弱支撑黄金', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: '数据源健康监控' })).toBeVisible();
  await expect(page.getByText('WGC Gold Demand')).toBeVisible();
  await expect(page.locator('.source-health-row').filter({ hasText: 'WGC Gold Demand' }).getByText('proxy_static_source')).toBeVisible();
  await expect(page.getByLabel('Market summary').getByText('XAUUSD', { exact: true })).toBeVisible();
  await expect(page.locator('.indicator-card-head').filter({ hasText: '基本面' })).toBeVisible();
  await expect(page.locator('.indicator-card-head').filter({ hasText: '技术面' })).toBeVisible();
  await expect(page.locator('.indicator-card-head').filter({ hasText: '宏观政策' })).toBeVisible();
  await expect(page.locator('.indicator-card-head').filter({ hasText: '资金情绪' })).toBeVisible();
  await expect(page.getByText('代理量化引擎：真实行情驱动')).toHaveCount(3);
  await expect(page.getByRole('link', { name: /进入风险画像 Agent/ })).toBeVisible();
});

test('dashboard exposes source audit details for each indicator', async ({ page }) => {
  await page.goto('/');

  await page.getByRole('button', { name: '审计 基本面A' }).click();

  await expect(page.getByRole('heading', { name: '指标证据审计' })).toBeVisible();
  await expect(page.getByText('来源 playwright-mock')).toBeVisible();
  await expect(page.getByText('状态 degraded')).toBeVisible();
  await expect(page.getByText('新鲜度 18s')).toBeVisible();
  await expect(page.getByText('降级原因 mock proxy')).toBeVisible();
  await expect(page.getByText('研究口径 指标用于解释市场基线，不直接改写预测价格。')).toBeVisible();
});

test('agent submits full investor profile and renders risk briefing', async ({ page }) => {
  await page.goto('/agent');

  await page.getByLabel('资金占比').fill('75');
  await page.getByLabel('最大回撤').fill('3');
  await page.getByLabel('杠杆态度').selectOption('high');
  await page.getByLabel('经验水平').selectOption('beginner');
  await page.getByLabel('已有持仓').selectOption('long');
  await expect(page.getByRole('heading', { name: '适当性门控预检' })).toBeVisible();
  await expect(page.getByText('强制观望')).toBeVisible();
  await expect(page.getByText('禁止加杠杆')).toBeVisible();
  await expect(page.getByRole('heading', { name: '风险预算计算器' })).toBeVisible();
  await expect(page.getByText('最大可承受亏损')).toBeVisible();
  await expect(page.locator('.position-mode-panel').getByText('已有多头', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '生成风险适配 briefing' }).click();

  await expect(page.getByRole('heading', { name: '风险适配分析' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '观望' })).toBeVisible();
  await expect(page.getByText('问卷风险门控')).toBeVisible();
  await expect(page.getByText('代理量化引擎：真实行情驱动')).toBeVisible();
  await expect(page.getByText('量化引擎当前不可用')).toHaveCount(0);
  await expect(page.getByRole('heading', { name: '用户风险画像' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '三情景执行框架' })).toBeVisible();
});

test('mobile dashboard does not create horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');

  await expect(page.getByRole('heading', { name: '黄金价格预测与指标总览' })).toBeVisible();
  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
  );
  expect(hasHorizontalOverflow).toBe(false);
});
