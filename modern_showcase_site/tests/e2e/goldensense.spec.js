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
  horizon_forecasts: ['short_term', 'mid_term', 'long_term'].map((horizon, index) => ({
    horizon,
    stance: index === 2 ? '中性' : '偏多',
    confidence_band: index === 0 ? '高' : '中',
    action: index === 2 ? '观望' : '观察确认',
    probability: index === 0 ? 0.68 : 0.61,
    basis: 'heuristic_proxy',
    model_status: horizon === 'long_term' ? 'not_applicable' : 'heuristic_proxy',
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
      event_id: 'fed-primary',
      published_at: '2026-04-30T07:45:00Z',
      title: 'Federal Reserve publishes policy statement',
      summary: 'Official policy release used as primary evidence.',
      source: 'Federal Reserve',
      normalized_event: 'fomc',
      sentiment_score: 0,
      importance: 1,
      categories: ['macro'],
      url: 'https://www.federalreserve.gov/newsevents/pressreleases/test.htm',
      source_tier: 'primary',
      source_authority: 'Federal Reserve',
      is_primary_source: true,
    },
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
      source_tier: 'secondary',
      source_authority: 'mock-wire',
      is_primary_source: false,
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
    horizon: 'short_term',
    confidence_band: '低',
    action: '观望',
    reasons: [
      '完整问卷触发风险画像门控：资金占比过高。',
      '短期当前采用代理预测，主要参考趋势、美元、利率和自动抓取的新闻环境。',
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

const researchCasePayload = {
  case_id: 'rc_playwright_closed_loop',
  question: 'FOMC statement impact on gold',
  asset: 'XAUUSD',
  created_at: '2026-08-13T08:00:00Z',
  data_asof: '2026-08-12',
  status: 'complete',
  research_mode: 'full',
  narrative: {
    overview: 'Accepted evidence was assembled into audited scenarios.',
    horizon_notes: {}, watchlist: [], generated_by: 'llm', degradation_flags: [],
  },
  investor_profile: null,
  evidence_documents: [{
    document_id: 'doc_mock', kind: 'url', filename: null, sha256: 'a'.repeat(64),
    source_url: 'https://www.federalreserve.gov/mock', content_type: 'text/html',
    source_tier: 'primary', retrieved_at: '2026-08-13T08:00:00Z', persisted_raw: false,
    extraction_status: 'complete', degradation_flags: [],
  }],
  fact_claims: [{
    claim_id: 'fact_mock', document_id: 'doc_mock', text: 'Real yields fell after the statement.',
    locator: 'https://www.federalreserve.gov/mock', status: 'accepted', confidence: 0.9, tags: [],
    domains: ['macro_event'],
  }],
  gate_report: [
    ['access', 'pass'], ['provenance_time', 'pass'], ['fact_location', 'pass'],
    ['ai_attack', 'pass'], ['dedup_replay', 'pass'], ['consistency', 'pass'],
    ['market_coherence', 'review'], ['output_audit', 'pass'],
  ].map(([gate, decision]) => ({
    gate, decision, reason: `${gate} mock audit`,
    confidence_multiplier: decision === 'review' ? 0.9 : 1, evidence_refs: ['fact_mock'],
  })),
  model_registry: [
    { model_id: 'flagship', label: 'Integrated flagship', state: 'production', role: 'champion', oos_metric: 'Sharpe 0.70 / max drawdown -21%', can_influence_strategy: true },
    { model_id: 'transformer', label: 'Transformer', state: 'watch', role: 'challenger', degradation_reason: 'not_walk_forward_validated', can_influence_strategy: false },
  ],
  conflicts: [{
    topic: 'macro vs flows', majority_view: 'no majority', minority_view: 'ETF flow disagrees',
    agent_ids: ['macro_event', 'technical_flows'], resolution: 'Preserve minority view.',
  }],
  agent_views: [{
    agent: 'macro_event', horizon: 'mid_term', stance: 'bullish', confidence: 0.68,
    thesis: 'Macro transmission view.', supporting_fact_ids: ['fact_mock'], counter_fact_ids: [],
    invalidation: ['real yields reverse'], degradation_flags: [],
    confidence_basis: { method: 'deterministic_evidence_score', calibrated: false, source_refs: ['fact_mock'] },
  }, {
    agent: 'quant_model_risk', horizon: 'mid_term', stance: 'bullish', confidence: 0.61,
    thesis: 'Governed model view.', supporting_fact_ids: [], counter_fact_ids: [],
    invalidation: ['governance fails'], degradation_flags: [],
    confidence_basis: { method: 'model_governance_score', calibrated: false, source_refs: ['macro_factors.composite'] },
  }],
  horizon_strategy: Object.fromEntries([
    ['short_term', 'risk'], ['mid_term', 'bullish'], ['long_term', 'neutral'],
  ].map(([horizon, stance]) => [horizon, {
    horizon, stance, confidence: 0.6, priced_in: 'uncertain',
    base: { label: 'base', probability: 0.5, description: `${horizon} base case`, probability_kind: 'research_weight', method: 'macro_composite_weight_v1' },
    upside: { label: 'upside', probability: 0.31, description: `${horizon} upside`, probability_kind: 'research_weight', method: 'macro_composite_weight_v1' },
    downside: { label: 'downside', probability: 0.19, description: `${horizon} downside`, probability_kind: 'research_weight', method: 'macro_composite_weight_v1' },
    triggers: ['real yields', 'USD reaction'], invalidation: ['market reaction reverses'],
    next_review_at: '2026-08-20T08:00:00Z', degradation_flags: horizon === 'short_term' ? ['unsupported_direction_model_direction_abstained'] : [],
    confidence_basis: { method: 'model_governance_score', calibrated: false, source_refs: ['model_registry'] },
    priced_in_basis: 'directional_alignment_proxy_not_market_reaction',
  }])),
  audit_report: { passed: true, issues: [], checked_at: '2026-08-13T08:00:00Z' },
  outcome_schedule: [{
    horizon: 'short_term', due_at: '2026-08-20T08:00:00Z', status: 'scored',
    entry_price: 3300, realized_price: 3366, realized_return: 0.02, direction_score: null,
    neutral_band_pct: 0.015, scenario_outcome: 'upside', scenario_score: 0.184,
    scenario_score_kind: 'weight_brier', confidence_error: null,
    scoring_method: 'horizon_band_multiclass_v1', scored_at: '2026-08-21T08:00:00Z',
  }, {
    horizon: 'mid_term', due_at: '2027-02-13T08:00:00Z', status: 'scored',
    entry_price: 3300, realized_price: 3300, realized_return: 0, direction_score: null,
    neutral_band_pct: 0.04, scenario_outcome: 'base', scenario_score: null,
    scenario_score_kind: null, confidence_error: null,
    scoring_method: 'horizon_band_multiclass_v1', scored_at: '2027-02-14T08:00:00Z',
  }],
};

const personalResearchPayload = {
  profile_echo: { risk_tolerance: 'balanced', horizon: 'mid', current_gold_pct: 10, experience: 'novice' },
  facts: {
    reference_range: { available: true, range_pct: [5, 15], midpoint: 10 },
    suitability: { status: 'insufficient', position_analysis_allowed: false, missing_fields: ['loss_capacity'], reasons: [], scope: 'education_only_unlevered_gold_research' },
    position_gap: { status: 'withheld', current_gold_pct: 10, gap_pct: null },
    risk_flags: [],
    horizon_evidence: { horizon: 'mid_term', section: { available: true, evidence: ['HMM elevated 70%'] } },
  },
  narrative: {
    overview: '研究画像摘要', position_analysis: '当前处于参考区间内。', risk_notes: [],
    horizon_note: '中期状态保持观察。', disclaimer: '仅用于教育型研究参考，不构成投资建议。',
  },
  degradation_flags: [], generated_by: 'deterministic_draft', mode: 'draft',
  research_case_id: 'rc_playwright_closed_loop',
  three_dimensional_brief: {
    case_id: 'rc_playwright_closed_loop',
    agent: { core_conclusion: '中期基础情景保持偏多观察。', scenario_focus: [] },
    rules: { risk_flags: [], hard_constraints: ['不输出直接买卖指令。'], suitability: { status: 'insufficient', position_analysis_allowed: false, missing_fields: ['loss_capacity'], reasons: [] } },
    api: { data_asof: '2026-08-12', freshness: 'current', model_states: researchCasePayload.model_registry },
    watchlist: ['real yields'], invalidation: ['market reaction reverses'], next_review_at: '2026-08-20T08:00:00Z',
  },
};

test.beforeEach(async ({ page }) => {
  await page.route('**/api/v1/agent/event-alert', async (route) => {
    await route.fulfill({ status: 404, json: { detail: 'no active event in fixture' } });
  });
  await page.route('**/api/v1/agent/market-view', async (route) => {
    const section = { available: true, core_view: 'Fixture view', confidence: '中', evidence: [], invalidation: ['Fixture invalidation'] };
    await route.fulfill({ json: { meta: { data_asof: '2026-08-12', data_age_days: 1, data_stale: false }, short_term: section, mid_term: section, long_term: section } });
  });
  await page.route('**/api/v1/agent/research/current', async (route) => {
    await route.fulfill({ status: 503, json: { detail: 'fixture uses active ResearchCase panels' } });
  });
  await page.route('**/api/v1/agent/calibration', async (route) => {
    await route.fulfill({ status: 503, json: { detail: 'no calibration fixture' } });
  });
  await page.route('**/api/v1/signals/current', async (route) => {
    await route.fulfill({ status: 404, json: { detail: 'empty fixture ledger' } });
  });
  await page.route('**/api/v1/signals/history?*', async (route) => {
    await route.fulfill({ json: { publications: [] } });
  });
  await page.route('**/api/v1/signals/track-record', async (route) => {
    await route.fulfill({ json: null });
  });
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
  await page.route('**/api/v1/agent/research-cases', async (route) => {
    await route.fulfill({ status: 201, json: researchCasePayload });
  });
  await page.route('**/api/v1/agent/research-cases/*', async (route) => {
    await route.fulfill({ json: researchCasePayload });
  });
  await page.route('**/api/v1/agent/personal-research?*', async (route) => {
    const url = new URL(route.request().url());
    const body = route.request().postDataJSON();
    const required = ['max_drawdown_pct', 'liquidity_need', 'leverage_attitude', 'investment_goal', 'loss_capacity', 'portfolio_context_known', 'emergency_fund_months', 'liabilities_level', 'gold_instrument', 'jurisdiction', 'base_currency'];
    const complete = required.every((key) => body[key] !== undefined && body[key] !== null);
    const suitability = complete
      ? { status: 'eligible', position_analysis_allowed: true, missing_fields: [], reasons: [] }
      : { status: 'insufficient', position_analysis_allowed: false, missing_fields: required.filter((key) => body[key] === undefined || body[key] === null), reasons: [] };
    await route.fulfill({
      json: {
        ...personalResearchPayload,
        mode: url.searchParams.get('mode') || 'full',
        facts: {
          ...personalResearchPayload.facts,
          suitability,
          position_gap: complete
            ? { status: 'within', current_gold_pct: 10, gap_pct: 0 }
            : { status: 'withheld', current_gold_pct: 10, gap_pct: null },
        },
        three_dimensional_brief: {
          ...personalResearchPayload.three_dimensional_brief,
          rules: { ...personalResearchPayload.three_dimensional_brief.rules, suitability },
        },
      },
    });
  });
});

async function usePro(page) {
  // These assertions cover methodology internals that simple mode folds away.
  await page.addInitScript(() => window.localStorage.setItem('gs_view_mode', 'pro'));
}

async function advanceAdvisorMobile(page, steps = 1) {
  if ((page.viewportSize()?.width || 1000) > 760) return;
  for (let current = 0; current < steps; current += 1) {
    await page.locator('.profile-mobile-actions button').last().click();
  }
}

async function fillSuitabilityProfile(page, { maxDrawdown = '12', leverage = 'none' } = {}) {
  await page.getByLabel('最大回撤承受力').fill(maxDrawdown);
  await page.getByLabel('流动性需求').selectOption('medium');
  await page.getByLabel('杠杆态度').selectOption(leverage);
  await page.getByLabel('投资目标').selectOption('capital_preservation');
  await page.getByLabel('损失承受能力').selectOption('medium');
  await page.getByLabel('组合上下文').selectOption('true');
  await page.getByLabel('应急资金月数').fill('9');
  await page.getByLabel('负债水平').selectOption('low');
  await page.getByLabel('黄金工具').selectOption('unlevered_etf');
  await page.getByLabel('运营法域').selectOption('SG');
  await page.getByLabel('基础货币').selectOption('SGD');
}

test('market-first dashboard orders live context before the research intake', async ({ page }) => {
  await page.goto('/');

  const sections = await page.locator('main [data-dashboard-section]').evaluateAll(
    (nodes) => nodes.map((node) => node.getAttribute('data-dashboard-section')),
  );
  expect(sections.indexOf('market-snapshot')).toBeLessThan(sections.indexOf('primary-news'));
  expect(sections.indexOf('primary-news')).toBeLessThan(sections.indexOf('research-summary'));
  expect(sections.indexOf('research-summary')).toBeLessThan(sections.indexOf('research-case'));
  await expect(page.getByRole('heading', { name: '一手信息' })).toBeVisible();
  await expect(page.getByText('官方原始来源')).toBeVisible();
  await expect(page.locator('[data-dashboard-section="primary-news"]').getByRole('heading', { name: 'Federal Reserve publishes policy statement' })).toBeVisible();
  await expect(page.getByText('补充背景')).toBeVisible();
});

test('dashboard failure has one honest state and a retry action', async ({ page }) => {
  await page.unroute('**/api/v1/agent/dashboard/current');
  await page.route('**/api/v1/agent/dashboard/current', async (route) => {
    await route.fulfill({ status: 503, json: { detail: 'RuntimeError: ConnectError' } });
  });
  await page.goto('/');

  await expect(page.getByRole('alert')).toContainText('暂时无法获取当前市场数据');
  await expect(page.getByRole('button', { name: '重新加载' })).toBeVisible();
  await expect(page.getByText('当前可用', { exact: true })).toHaveCount(0);
  await expect(page.getByText('质量提示', { exact: true })).toHaveCount(0);
  await expect(page.getByText(/RuntimeError|ConnectError/)).toHaveCount(0);
});

test('incomplete profile is not assigned a synthetic risk score', async ({ page }) => {
  await page.addInitScript(() => window.localStorage.removeItem('gs_profile_v2'));
  await page.goto('/advisor');

  await expect(page.getByText('画像未完成', { exact: true })).toBeVisible();
  await expect(page.getByText(/score\s+\d/i)).toHaveCount(0);
  await expect(page.getByText(/风险分\s*\d/)).toHaveCount(0);
  await expect(page.getByText('第 1 步，共 3 步')).toBeVisible();
  await expect(page.getByText('暂不计算个人风险等级')).toBeVisible();
  await advanceAdvisorMobile(page, 2);
  await page.getByRole('tab', { name: /^提问分析/ }).click();
  await expect(page.locator('.gate-decision-row')).toContainText('画像未完成');
  await expect(page.locator('.gate-decision-row')).not.toContainText(/风险分\s*\d/);
});

test('mobile navigation stays compact and touch friendly', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');

  await expect(page.getByRole('navigation', { name: '移动端主导航' })).toBeVisible();
  await expect(page.locator('.topnav')).toBeHidden();
  const heights = await page.getByRole('navigation', { name: '移动端主导航' })
    .getByRole('link').evaluateAll((links) => links.map((link) => link.getBoundingClientRect().height));
  expect(heights.every((height) => height >= 44)).toBe(true);
});

test('mobile primary controls and evidence links meet 44px touch targets', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');

  const homeTargets = page.locator('.case-input-tabs button, .case-run-button, .source-news-list a');
  const homeHeights = await homeTargets.evaluateAll((nodes) => nodes
    .filter((node) => node.getBoundingClientRect().width > 0)
    .map((node) => node.getBoundingClientRect().height));
  expect(homeHeights.length).toBeGreaterThan(0);
  expect(homeHeights.every((height) => height >= 44)).toBe(true);

  await page.goto('/signals');
  const subscribeTargets = page.locator('.subscribe-row input, .subscribe-row button');
  const subscribeHeights = await subscribeTargets.evaluateAll((nodes) => nodes
    .filter((node) => node.getBoundingClientRect().width > 0)
    .map((node) => node.getBoundingClientRect().height));
  expect(subscribeHeights.length).toBe(2);
  expect(subscribeHeights.every((height) => height >= 44)).toBe(true);
  expect(await page.locator('body').evaluate((body) => body.scrollWidth <= window.innerWidth)).toBe(true);
});

test('search traps focus and returns it to the trigger', async ({ page }) => {
  await page.goto('/');
  const trigger = page.getByRole('button', { name: '搜索研究内容' });
  await trigger.click();
  await expect(page.getByRole('dialog', { name: '全局搜索' })).toBeVisible();
  // Dashboard/news search entries can arrive just after the dialog opens;
  // wait for that one controlled rerender before testing the boundary node.
  await page.waitForTimeout(300);
  const lastResult = page.locator('.search-result').last();
  await lastResult.focus();
  await page.keyboard.press('Tab');
  await expect(page.getByLabel('搜索', { exact: true })).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(trigger).toBeFocused();
});

test('quant credibility abstains on implausible values and removes duplicate outcomes', async ({ page }) => {
  await page.unroute('**/api/v1/agent/research/current');
  await page.route('**/api/v1/agent/research/current', async (route) => {
    await route.fulfill({ json: {
      data_asof: '2026-08-12', data_age_days: 1, data_stale: false, data_source: 'fixture',
      data_span: ['2004-01-01', '2026-08-12'], data_rows: 5000, degraded: {},
      fair_value: {
        deviation_pct: 69.8, deviation_z: 4.2, fair_value: 1950, spot: 3311,
        r_squared: 0.01, half_life_days: 53772, regime_break: false,
        interpretation: '关系正常', deviation_series_tail: [60, 69.8], quartile_forward_returns: {},
      },
    } });
  });
  await page.unroute('**/api/v1/agent/calibration');
  await page.route('**/api/v1/agent/calibration', async (route) => {
    const outcome = { analysis_id: 'duplicate', hit: true, stance: '偏多', horizon: 'mid_term', realized_return: 0.01, created_at: '2026-08-12' };
    await route.fulfill({ json: { total_scored: 2, hit_rate: 1, brier_score: 0.1, neutral_or_gated: 0, recent_outcomes: [outcome, outcome] } });
  });
  await page.goto('/quant');

  await expect(page.getByText('未观察到有效均值回归')).toBeVisible();
  await expect(page.getByText('53772 日')).toHaveCount(0);
  await expect(page.locator('.calib-item')).toHaveCount(1);
  await expect(page.getByText('估值关系显著偏离')).toBeVisible();
});

test('simple quant mode uses plain language instead of model abbreviations', async ({ page }) => {
  await page.goto('/quant');
  await expect(page.locator('body')).not.toContainText(/HAR-RV|HMM|BL-lite|Brier|OOS/);
});

test('dashboard presents forecasts and four indicator pillars', async ({ page }) => {
  await usePro(page);
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
  await usePro(page);
  await page.goto('/');

  await page.getByRole('button', { name: '审计 基本面A' }).click();

  await expect(page.getByRole('heading', { name: '指标证据审计' })).toBeVisible();
  await expect(page.getByText('来源 playwright-mock')).toBeVisible();
  await expect(page.getByText('状态 degraded')).toBeVisible();
  await expect(page.getByText('新鲜度 18s')).toBeVisible();
  await expect(page.getByText('降级原因 mock proxy')).toBeVisible();
  await expect(page.getByText('研究口径 指标用于解释市场基线，不直接改写预测价格。')).toBeVisible();
});

test('unified workbench: shared profile + Q&A analysis renders risk briefing', async ({ page }) => {
  await page.goto('/agent'); // alias of the unified /advisor workbench

  // Shared profile card (visible on both capability tabs).
  await page.getByLabel('当前黄金仓位').fill('75');
  await page.getByRole('button', { name: '新手', exact: true }).click();
  await advanceAdvisorMobile(page);
  await fillSuitabilityProfile(page, { maxDrawdown: '3', leverage: 'high' });
  await advanceAdvisorMobile(page);

  // Switch to the 提问分析 capability.
  await page.getByRole('tab', { name: /提问分析/ }).click();
  await expect(page.getByRole('heading', { name: '适当性门控预检' })).toBeVisible();
  await expect(page.getByText('强制观望')).toBeVisible();
  await expect(page.getByText('禁止加杠杆')).toBeVisible();
  await expect(page.getByRole('heading', { name: '风险预算画像' })).toBeVisible();
  await expect(page.getByText('声明回撤承受力')).toBeVisible();
  // A 75% gold position maps to a long position in the legacy contract.
  await expect(page.locator('.position-mode-panel').getByText('已有多头', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '生成风险适配分析' }).click();

  await expect(page.getByRole('heading', { name: '风险适配分析' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '观望' })).toBeVisible();
  await expect(page.getByText('问卷风险门控')).toBeVisible();
  await expect(page.getByText('代理量化引擎：真实行情驱动')).toBeVisible();
  await expect(page.getByText('量化引擎当前不可用')).toHaveCount(0);
  await expect(page.getByRole('heading', { name: '用户风险画像' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '三情景执行框架' })).toBeVisible();
  await expect(page.getByText(/(?:加仓|减仓|仓试探|立即买入|立即卖出)/)).toHaveCount(0);
});

test('unified workbench: allocation research tab is the default capability', async ({ page }) => {
  await page.goto('/advisor');
  await expect(page.getByRole('heading', { name: '个性化投研' })).toBeVisible();
  await advanceAdvisorMobile(page, 2);
  // Default tab = 配置研究, with its generate action.
  await expect(page.getByRole('button', { name: /生成个性化配置研究/ })).toBeVisible();
  // Both capability tabs are present.
  await expect(page.getByRole('tab', { name: /配置研究/ })).toBeVisible();
  await expect(page.getByRole('tab', { name: /提问分析/ })).toBeVisible();
});

test('simple mode hides methodology internals until toggled', async ({ page }) => {
  await page.goto('/');

  // Default is simple: audits and source health are folded away...
  await expect(page.getByRole('heading', { name: '今日核心结论' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '数据源健康监控' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '审计 基本面A' })).toHaveCount(0);
  await expect(page.getByText('简明模式已折叠逐项指标审计')).toBeVisible();

  // ...and the topbar toggle reveals them without a reload.
  await page.getByRole('button', { name: '专业', exact: true }).click();
  await expect(page.getByRole('heading', { name: '数据源健康监控' })).toBeVisible();
  await expect(page.getByRole('button', { name: '审计 基本面A' })).toBeVisible();
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

test('one research case connects intake, shield, models, strategy and personalization', async ({ page }) => {
  await usePro(page);
  await page.goto('/');

  await expect(page.getByRole('heading', { name: '创建统一研究案件' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'URL' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'PDF' })).toBeVisible();
  await expect(page.getByRole('button', { name: '图片' })).toBeVisible();
  await page.getByLabel('研究问题').fill('FOMC statement impact on gold');
  await page.getByLabel('证据 URL').fill('https://www.federalreserve.gov/mock');
  await page.getByRole('button', { name: '运行研究闭环' }).click();

  await expect(page.getByText('rc_playwright_closed_loop', { exact: true })).toBeVisible();
  await expect(page.getByText('Full · LLM叙事已审计')).toBeVisible();
  const persistedReference = await page.evaluate(() => (
    JSON.parse(window.localStorage.getItem('gs_active_research_case_v1'))
  ));
  expect(persistedReference).toEqual({ case_id: 'rc_playwright_closed_loop' });
  await expect(page.getByRole('heading', { name: 'Evidence Shield · 8层证据护盾' })).toBeVisible();
  await expect(page.getByText('AI攻击门')).toBeVisible();

  await page.getByRole('link', { name: /量化研究/ }).click();
  await expect(page.getByText('rc_playwright_closed_loop', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: '模型冠军—挑战者治理' })).toBeVisible();
  await expect(page.getByText('Transformer')).toBeVisible();
  await expect(page.getByText('观察', { exact: true })).toBeVisible();

  await page.getByRole('link', { name: /观点书与台账/ }).click();
  await expect(page.getByText('历史回测', { exact: true })).toBeVisible();
  await expect(page.getByText('模拟前向', { exact: true })).toBeVisible();
  await expect(page.getByText('真实前向', { exact: true })).toBeVisible();
  await expect(page.getByText('需要第一期真实前向发布')).toBeVisible();
  await expect(page.getByRole('heading', { name: '当前案件三期限策略' })).toBeVisible();
  await expect(page.getByText('少数意见保留')).toBeVisible();
  await expect(page.getByText('研究权重 · 非校准概率').first()).toBeVisible();
  await expect(page.getByText('macro_composite_weight_v1').first()).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Agent 专属证据账本' })).toBeVisible();
  await expect(page.getByText('事件宏观 Agent')).toBeVisible();
  await expect(page.getByText('权重 Brier 0.184')).toBeVisible();
  await expect(page.getByText('已到期 · 评分不可用')).toBeVisible();
  await expect(page.getByText('权重 Brier 0.000')).toHaveCount(0);

  await page.getByRole('link', { name: /个性化投研/ }).click();
  await expect(page.getByText('rc_playwright_closed_loop', { exact: true })).toBeVisible();
  await advanceAdvisorMobile(page, 2);
  await page.getByRole('button', { name: /生成个性化配置研究/ }).click();
  await expect(page.getByRole('heading', { name: 'Agent × 规则 × API 三维建议' })).toBeVisible();
  await expect(page.getByText('Agent解释层')).toBeVisible();
  await expect(page.getByText('硬规则层')).toBeVisible();
  await expect(page.getByText('实时API层')).toBeVisible();
  await expect(page.getByText('适当性信息不足')).toBeVisible();
  await expect(page.getByText('个人仓位差距已暂停')).toBeVisible();
});

test('complete suitability profile unlocks position comparison without trade directives', async ({ page }) => {
  await page.goto('/advisor');
  await advanceAdvisorMobile(page);
  await fillSuitabilityProfile(page);
  await advanceAdvisorMobile(page);
  await page.getByRole('button', { name: /生成个性化配置研究/ }).click();

  await expect(page.getByText('适当性通过')).toBeVisible();
  await expect(page.getByText('允许显示个人仓位差距')).toBeVisible();
  await expect(page.getByText('立即买入')).toHaveCount(0);
});
