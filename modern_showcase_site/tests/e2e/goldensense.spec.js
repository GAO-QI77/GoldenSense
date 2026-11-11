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

const productionEmptySignalPayload = {
  status: 'empty',
  error_code: 'first_publication_required',
  guidance: '由内部发布任务冻结本周研究。',
};

const productionWithheldPersonalResearchPayload = {
  ...personalResearchPayload,
  profile_echo: {
    ...personalResearchPayload.profile_echo,
    current_gold_pct: 60,
  },
  facts: {
    ...personalResearchPayload.facts,
    reference_range: { available: true, range_pct: [4, 9.1], midpoint: 6.55 },
    position_gap: { status: 'withheld', current_gold_pct: 60, gap_pct: null },
    suitability: {
      status: 'insufficient',
      position_analysis_allowed: false,
      missing_fields: ['loss_capacity'],
      reasons: [],
      scope: 'education_only_unlevered_gold_research',
    },
  },
  narrative: {
    ...personalResearchPayload.narrative,
    position_analysis: '个人仓位差距暂不计算。',
  },
  three_dimensional_brief: {
    ...personalResearchPayload.three_dimensional_brief,
    rules: {
      ...personalResearchPayload.three_dimensional_brief.rules,
      suitability: {
        status: 'insufficient',
        position_analysis_allowed: false,
        missing_fields: ['loss_capacity'],
        reasons: [],
      },
    },
  },
};

const productionCalibrationAccumulationPayload = {
  total_scored: 0,
  directional_calls: 0,
  minimum_directional_calls: 20,
  pending_analyses: 1,
  evidence_status: 'accumulating',
  hit_rate: null,
  brier_score: null,
  governance: {
    mode: 'insufficient_data',
    directional_calls: 0,
    min_directional_calls: 20,
  },
};

const productionSourceHealthRows = [
  {
    id: 'static_proxy',
    label: 'Static proxy source',
    source_type: 'fundamental',
    status: 'degraded',
    freshness_seconds: 11,
    expected_lag_seconds: 180,
    cadence: '日内',
    degraded_reason: 'proxy_static_source',
    coverage: ['黄金需求'],
    url: null,
  },
  {
    id: 'real_yield_proxy',
    label: 'Real yield proxy',
    source_type: 'macro_policy',
    status: 'degraded',
    freshness_seconds: 11,
    expected_lag_seconds: 180,
    cadence: '日内',
    degraded_reason: 'real_yield_proxy_from_nominal_yield',
    coverage: ['实际利率'],
    url: null,
  },
];

test.beforeEach(async ({ page }) => {
  await page.route('**/api/v1/agent/event-alert', async (route) => {
    await route.fulfill({ status: 404, json: { detail: 'no active event in fixture' } });
  });
  await page.route('**/api/v1/agent/research/current', async (route) => {
    await route.fulfill({ status: 503, json: { detail: 'fixture uses active ResearchCase panels' } });
  });
  await page.route('**/api/v1/agent/calibration', async (route) => {
    await route.fulfill({ status: 503, json: { detail: 'no calibration fixture' } });
  });
  await page.route('**/api/v1/signals/current', async (route) => {
    await route.fulfill({ status: 200, json: productionEmptySignalPayload });
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
  await page.route('**/api/v1/agent/market-view', async (route) => {
    await route.fulfill({ json: {
      meta: { data_asof: '2026-08-12', data_age_days: 1, data_stale: false },
      short_term: { available: true, confidence: '中', core_view: '短期关注风险分布。' },
      mid_term: { available: true, confidence: '中', core_view: '中期关注宏观驱动。' },
      long_term: { available: true, confidence: '中', core_view: '长期关注估值与配置。' },
    } });
  });
  await page.route('**/api/v1/agent/analyze', async (route) => {
    const body = route.request().postDataJSON();
    expect(body.investor_profile).toBeTruthy();
    expect(body.investor_profile.capital_allocation_pct).toBeGreaterThanOrEqual(0);
    await route.fulfill({
      json: analysisPayload,
      headers: {
        'X-Analysis-Feedback-Token': 'test-analysis-feedback-token',
        'Access-Control-Expose-Headers': 'X-Analysis-Feedback-Token',
      },
    });
  });
  await page.route('**/api/v1/agent/feedback', async (route) => {
    expect(route.request().headers()['x-analysis-feedback-token']).toBe('test-analysis-feedback-token');
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

async function computedContrast(locator) {
  await locator.scrollIntoViewIfNeeded();
  const measurement = await locator.evaluate((element) => {
    const elementRect = element.getBoundingClientRect();
    if (!elementRect.width || !elementRect.height) {
      throw new Error('Contrast target has no painted area');
    }

    const maxTextRects = 24;
    const xFractions = [0.05, 0.25, 0.5, 0.75, 0.95];
    const yFractions = [0.2, 0.5, 0.8];
    const textRects = [];
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    const range = document.createRange();
    for (let textNode = walker.nextNode(); textNode; textNode = walker.nextNode()) {
      if (!textNode.textContent?.trim()) continue;
      range.selectNodeContents(textNode);
      textRects.push(...range.getClientRects());
    }
    if (!textRects.length) throw new Error('Contrast target has no painted text');
    if (textRects.length > maxTextRects) {
      throw new Error(`Contrast target has ${textRects.length} text rects; maximum is ${maxTextRects}`);
    }

    const sampleKeys = new Set();
    const samples = [];
    for (const textRect of textRects) {
      const left = Math.max(elementRect.left, textRect.left);
      const right = Math.min(elementRect.right, textRect.right);
      const top = Math.max(elementRect.top, textRect.top);
      const bottom = Math.min(elementRect.bottom, textRect.bottom);
      if (right <= left || bottom <= top) continue;
      for (const xFraction of xFractions) {
        for (const yFraction of yFractions) {
          const xRatio = (left + (right - left) * xFraction - elementRect.left) / elementRect.width;
          const yRatio = (top + (bottom - top) * yFraction - elementRect.top) / elementRect.height;
          const key = `${xRatio.toFixed(5)}:${yRatio.toFixed(5)}`;
          if (sampleKeys.has(key)) continue;
          sampleKeys.add(key);
          samples.push({ xRatio, yRatio });
        }
      }
    }
    if (!samples.length) throw new Error('Contrast target text is outside its painted area');

    const probeSize = 6;
    const probeBatches = [];
    samples.forEach((sample, sampleIndex) => {
      const x = sample.xRatio * elementRect.width;
      const y = sample.yRatio * elementRect.height;
      let batch = probeBatches.find((candidate) => candidate.every((otherIndex) => {
        const other = samples[otherIndex];
        const otherX = other.xRatio * elementRect.width;
        const otherY = other.yRatio * elementRect.height;
        return Math.abs(x - otherX) >= probeSize || Math.abs(y - otherY) >= probeSize;
      }));
      if (!batch) {
        batch = [];
        probeBatches.push(batch);
      }
      batch.push(sampleIndex);
    });

    const inlineStyle = {
      color: element.style.getPropertyValue('color'),
      colorPriority: element.style.getPropertyPriority('color'),
      textShadow: element.style.getPropertyValue('text-shadow'),
      textShadowPriority: element.style.getPropertyPriority('text-shadow'),
      textFill: element.style.getPropertyValue('-webkit-text-fill-color'),
      textFillPriority: element.style.getPropertyPriority('-webkit-text-fill-color'),
    };
    const foregroundCss = getComputedStyle(element).color;
    element.style.setProperty('color', 'transparent', 'important');
    element.style.setProperty('text-shadow', 'none', 'important');
    element.style.setProperty('-webkit-text-fill-color', 'transparent', 'important');

    return {
      foregroundCss,
      inlineStyle,
      probeBatches,
      samples,
    };
  });

  let backgroundScreenshot;
  const foregroundScreenshots = [];
  try {
    backgroundScreenshot = await locator.screenshot({ animations: 'disabled', caret: 'hide' });
    for (const sampleIndexes of measurement.probeBatches) {
      await locator.evaluate((element, { foregroundCss, sampleIndexes, samples }) => {
        const elementRect = element.getBoundingClientRect();
        for (const sampleIndex of sampleIndexes) {
          const sample = samples[sampleIndex];
          const probe = document.createElement('i');
          probe.dataset.contrastProbe = '';
          probe.setAttribute('aria-hidden', 'true');
          Object.assign(probe.style, {
            position: 'fixed',
            left: `${elementRect.left + sample.xRatio * elementRect.width - 3}px`,
            top: `${elementRect.top + sample.yRatio * elementRect.height - 3}px`,
            width: '6px',
            height: '6px',
            display: 'block',
            margin: '0',
            padding: '0',
            border: '0',
            borderRadius: '0',
            background: foregroundCss,
            boxShadow: 'none',
            pointerEvents: 'none',
            zIndex: '2147483647',
          });
          element.appendChild(probe);
        }
      }, {
        foregroundCss: measurement.foregroundCss,
        sampleIndexes,
        samples: measurement.samples,
      });
      foregroundScreenshots.push({
        sampleIndexes,
        screenshot: await locator.screenshot({ animations: 'disabled', caret: 'hide' }),
      });
      await locator.evaluate((element) => {
        for (const probe of element.querySelectorAll('[data-contrast-probe]')) probe.remove();
      });
    }
  } finally {
    await locator.evaluate((element, inlineStyle) => {
      for (const probe of element.querySelectorAll('[data-contrast-probe]')) probe.remove();
      const restore = (property, value, priority) => {
        if (value) element.style.setProperty(property, value, priority);
        else element.style.removeProperty(property);
      };
      restore('color', inlineStyle.color, inlineStyle.colorPriority);
      restore('text-shadow', inlineStyle.textShadow, inlineStyle.textShadowPriority);
      restore('-webkit-text-fill-color', inlineStyle.textFill, inlineStyle.textFillPriority);
    }, measurement.inlineStyle);
  }

  const paintedColors = await locator.page().evaluate(async ({ backgroundUrl, foregroundBatches, samples }) => {
    const sampleScreenshot = async (dataUrl, requestedSamples) => {
      const response = await fetch(dataUrl);
      const bitmap = await createImageBitmap(await response.blob());
      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      context.drawImage(bitmap, 0, 0);
      const colors = requestedSamples.map((sample) => {
        const x = Math.min(bitmap.width - 1, Math.max(0, Math.floor(sample.xRatio * bitmap.width)));
        const y = Math.min(bitmap.height - 1, Math.max(0, Math.floor(sample.yRatio * bitmap.height)));
        const [r, g, b, alphaByte] = context.getImageData(x, y, 1, 1).data;
        return { r, g, b, a: alphaByte / 255 };
      });
      bitmap.close();
      return colors;
    };
    const foregrounds = Array(samples.length);
    for (const batch of foregroundBatches) {
      const colors = await sampleScreenshot(
        batch.dataUrl,
        batch.sampleIndexes.map((sampleIndex) => samples[sampleIndex]),
      );
      batch.sampleIndexes.forEach((sampleIndex, index) => {
        foregrounds[sampleIndex] = colors[index];
      });
    }
    return {
      backgrounds: await sampleScreenshot(backgroundUrl, samples),
      foregrounds,
    };
  }, {
    backgroundUrl: `data:image/png;base64,${backgroundScreenshot.toString('base64')}`,
    foregroundBatches: foregroundScreenshots.map(({ sampleIndexes, screenshot }) => ({
      dataUrl: `data:image/png;base64,${screenshot.toString('base64')}`,
      sampleIndexes,
    })),
    samples: measurement.samples,
  });

  const composite = (foreground, backdrop) => ({
    r: foreground.r * foreground.a + backdrop.r * (1 - foreground.a),
    g: foreground.g * foreground.a + backdrop.g * (1 - foreground.a),
    b: foreground.b * foreground.a + backdrop.b * (1 - foreground.a),
  });
  const luminance = ({ r, g, b }) => {
    const linear = [r, g, b].map((channel) => {
      const normalized = channel / 255;
      return normalized <= 0.04045
        ? normalized / 12.92
        : ((normalized + 0.055) / 1.055) ** 2.4;
    });
    return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
  };
  return Math.min(...paintedColors.backgrounds.map((background, index) => {
    const paintedBackground = composite(background, { r: 255, g: 255, b: 255 });
    const paintedForeground = composite(paintedColors.foregrounds[index], { r: 255, g: 255, b: 255 });
    const foregroundLuminance = luminance(paintedForeground);
    const backgroundLuminance = luminance(paintedBackground);
    return (
      (Math.max(foregroundLuminance, backgroundLuminance) + 0.05) /
      (Math.min(foregroundLuminance, backgroundLuminance) + 0.05)
    );
  }));
}

async function expectAccessibleLightCardText(locator) {
  await expect(locator).toBeVisible();
  expect(await computedContrast(locator)).toBeGreaterThanOrEqual(4.5);
  const isNearWhite = await locator.evaluate((element) => {
    const channels = getComputedStyle(element).color.match(/[\d.]+/g)?.map(Number) || [];
    return channels.length >= 3 && channels.slice(0, 3).every((channel) => channel >= 220);
  });
  expect(isNearWhite).toBe(false);
}

test('contrast helper composites translucent text over its painted background', async ({ page }) => {
  await page.setContent(`
    <div style="padding: 24px; background: white;">
      <span id="translucent-text" style="color: rgba(0, 0, 0, 0.2);">translucent text</span>
    </div>
  `);

  const contrast = await computedContrast(page.locator('#translucent-text'));
  expect(contrast).toBeCloseTo(1.61, 2);
  expect(contrast).toBeLessThan(4.5);
});

test('contrast helper samples narrow translucent text without overlapping probes', async ({ page }) => {
  await page.setContent(`
    <div style="padding: 24px; background: white;">
      <span id="narrow-translucent-text" style="color: rgba(0, 0, 0, 0.4);">i</span>
    </div>
  `);

  const contrast = await computedContrast(page.locator('#narrow-translucent-text'));
  expect(contrast).toBeCloseTo(2.85, 2);
  expect(contrast).toBeLessThan(4.5);
});

test('contrast helper applies ancestor group opacity once to text and background', async ({ page }) => {
  await page.setContent(`
    <div style="padding: 24px; background: white;">
      <div style="padding: 12px; background: rgb(120, 120, 120); opacity: 0.7;">
        <span id="group-opacity-text" style="color: black;">group opacity text</span>
      </div>
    </div>
  `);

  const contrast = await computedContrast(page.locator('#group-opacity-text'));
  expect(contrast).toBeCloseTo(3.28, 1);
  expect(contrast).toBeLessThan(4.5);
});

test('contrast helper reports the worst region across gradient-backed text lines', async ({ page }) => {
  await page.setContent(`
    <div
      id="gradient-text"
      style="box-sizing: border-box; width: 360px; padding: 12px; color: black;
             background: linear-gradient(to right, white 0%, white 50%, black 80%, black 100%);
             font: 16px/20px monospace; white-space: nowrap;"
    >safe<br>0123456789012345678901234567890123</div>
  `);

  const textLineCount = await page.locator('#gradient-text').evaluate((element) => {
    const range = document.createRange();
    range.selectNodeContents(element);
    return range.getClientRects().length;
  });
  expect(textLineCount).toBeGreaterThan(1);

  const contrast = await computedContrast(page.locator('#gradient-text'));
  expect(contrast).toBeLessThan(1.1);
  expect(contrast).toBeLessThan(4.5);
});

test.describe('reported production screenshots', () => {
  test('production state: typed empty signal stays date-safe and user-readable', async ({ page }) => {
    await page.goto('/signals');

    const currentLedger = page.locator('.ledger-current');
    await expect(currentLedger).toContainText('需要第一期真实前向发布');
    await expect(currentLedger).toContainText('由内部发布任务冻结本周研究。');
    await expect(page.locator('body')).not.toContainText('Invalid Date');
    await expect(page.locator('body')).not.toContainText('first_publication_required');
    await expectAccessibleLightCardText(currentLedger.locator('.ledger-empty-guidance p'));
    await expectAccessibleLightCardText(currentLedger.locator('.ledger-empty-guidance small'));
  });

  test('withheld production personal research keeps the valid range and accessible contrast', async ({ page }) => {
    await page.unroute('**/api/v1/agent/personal-research?*');
    await page.route('**/api/v1/agent/personal-research?*', async (route) => {
      await route.fulfill({ json: productionWithheldPersonalResearchPayload });
    });

    await page.goto('/advisor');
    await advanceAdvisorMobile(page, 2);
    await page.getByRole('button', { name: /生成个性化配置研究/ }).click();

    const rangePanel = page.locator('.advisor-range-panel');
    await expect(rangePanel).toContainText('4.0% – 9.1%');
    await expect(rangePanel).toContainText('中点 6.55%');
    await expect(rangePanel).toContainText('参考区间可用；个人仓位差距暂不计算。');
    await expect(rangePanel).toContainText('当前仓位 60%');
    await expect(rangePanel).toContainText('个人仓位差距已暂停');
    await expect(page.locator('body')).not.toContainText('Invalid Date');
    await expect(page.locator('body')).not.toContainText('withheld');
    await expectAccessibleLightCardText(rangePanel.locator('.advisor-gap-line > span').last());
    await expectAccessibleLightCardText(rangePanel.locator('.suitability-verdict span'));
    await expectAccessibleLightCardText(rangePanel.locator('.suitability-verdict strong'));
  });

  test('calibration accumulation reports 0/20 without mature score tiles', async ({ page }) => {
    await page.unroute('**/api/v1/agent/research/current');
    await page.route('**/api/v1/agent/research/current', async (route) => {
      await route.fulfill({ json: {
        data_asof: '2026-08-12',
        data_age_days: 1,
        data_stale: false,
        data_source: 'fixture',
        data_span: ['2004-01-01', '2026-08-12'],
        data_rows: 5000,
        degraded: {},
      } });
    });
    await page.unroute('**/api/v1/agent/calibration');
    await page.route('**/api/v1/agent/calibration', async (route) => {
      await route.fulfill({ json: productionCalibrationAccumulationPayload });
    });

    await page.goto('/quant');

    const calibrationPanel = page.locator('#panel-calibration');
    await expect(calibrationPanel.getByText('正常积累中 0/20')).toBeVisible();
    await expect(calibrationPanel.getByText('另有 1 条分析等待复核日成熟。')).toBeVisible();
    await expect(calibrationPanel.locator('.calib-tiles')).toHaveCount(0);
    await expect(calibrationPanel.getByText('已评分判断')).toHaveCount(0);
    await expect(calibrationPanel.getByText('方向性判断命中率')).toHaveCount(0);
    await expect(page.locator('body')).not.toContainText('insufficient_data');
    await expect(page.locator('body')).not.toContainText('Invalid Date');
  });

  test('production state source health separates on-time provenance with accessible contrast', async ({ page }) => {
    await page.unroute('**/api/v1/agent/dashboard/current');
    await page.route('**/api/v1/agent/dashboard/current', async (route) => {
      await route.fulfill({ json: {
        ...dashboardPayload,
        source_health: productionSourceHealthRows,
      } });
    });

    await page.goto('/');
    await expect(page.locator('body')).not.toContainText('proxy_static_source');
    await expect(page.locator('body')).not.toContainText('real_yield_proxy_from_nominal_yield');
    await page.getByRole('button', { name: '专业', exact: true }).click();

    const staticProxy = page.locator('.source-health-row').filter({ hasText: 'Static proxy source' });
    const realYieldProxy = page.locator('.source-health-row').filter({ hasText: 'Real yield proxy' });
    await expect(staticProxy.getByText('时效 · 按时', { exact: true })).toBeVisible();
    await expect(staticProxy.getByText('出处 · 代理值', { exact: true })).toBeVisible();
    await expect(staticProxy).not.toContainText('时效 · 延迟');
    await expect(staticProxy.locator('p')).toContainText('当前使用代理数据，并非实时官方数据');
    await expect(realYieldProxy.getByText('时效 · 按时', { exact: true })).toBeVisible();
    await expect(realYieldProxy.getByText('出处 · 衍生值', { exact: true })).toBeVisible();
    await expect(realYieldProxy).not.toContainText('时效 · 延迟');
    await expect(realYieldProxy.locator('p')).toContainText('实际利率由名义利率估算，属于衍生代理值');
    await expectAccessibleLightCardText(staticProxy.locator('p'));
    await expectAccessibleLightCardText(staticProxy.locator('.source-health-meta span').first());
    await expectAccessibleLightCardText(staticProxy.getByText('出处 · 代理值', { exact: true }));
    await expectAccessibleLightCardText(realYieldProxy.locator('p'));
    await expectAccessibleLightCardText(realYieldProxy.getByText('出处 · 衍生值', { exact: true }));
    await expect(page.locator('body')).not.toContainText('Invalid Date');
  });
});

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

test('signals current outage preserves healthy market view without inventing a publication', async ({ page }) => {
  await page.unroute('**/api/v1/signals/current');
  await page.route('**/api/v1/signals/current', async (route) => {
    await route.fulfill({ status: 503, json: { detail: 'publication store unavailable' } });
  });

  await page.goto('/signals');

  await expect(page.getByText('短期关注风险分布。', { exact: true })).toBeVisible();
  await expect(page.getByText('观点书不可用')).toHaveCount(0);
  await expect(page.getByText('本周发布暂不可用')).toBeVisible();
  await expect(page.getByText('等待第一期发布，不回填历史。')).toHaveCount(0);
});

test('market-view outage still settles healthy ledger modules', async ({ page }) => {
  const publication = {
    publication_id: '2026-W36',
    published_at: '2026-09-03T01:00:00Z',
    data_asof: '2026-09-02',
    allocations: {
      conservative: { range_pct: [2, 8], midpoint: 5 },
      balanced: { range_pct: [5, 12], midpoint: 8.5 },
      aggressive: { range_pct: [8, 18], midpoint: 13 },
    },
    content_hash: 'sha256:fixture-publication-hash',
    disclaimer: 'Fixture live-forward publication.',
  };
  const trackRecord = {
    matured_through: '2026-W35',
    cost_bps: 5,
    per_profile: {
      conservative: { weeks_scored: 1, cum_return: 0.01, max_drawdown: -0.005 },
      balanced: { weeks_scored: 1, cum_return: 0.015, max_drawdown: -0.006 },
      aggressive: { weeks_scored: 1, cum_return: 0.02, max_drawdown: -0.008 },
    },
    benchmarks: { gold_buy_hold: { cum_return: 0.012 } },
    disclaimer: 'Fixture simulated-forward scorecard.',
  };

  await usePro(page);
  await page.unroute('**/api/v1/agent/market-view');
  await page.unroute('**/api/v1/signals/current');
  await page.unroute('**/api/v1/signals/history?*');
  await page.unroute('**/api/v1/signals/track-record');
  await page.route('**/api/v1/agent/market-view', async (route) => {
    await route.fulfill({ status: 503, json: { detail: 'market view unavailable' } });
  });
  await page.route('**/api/v1/signals/current', async (route) => {
    await route.fulfill({ json: publication });
  });
  await page.route('**/api/v1/signals/history?*', async (route) => {
    await route.fulfill({ json: { publications: [publication] } });
  });
  await page.route('**/api/v1/signals/track-record', async (route) => {
    await route.fulfill({ json: trackRecord });
  });

  await page.goto('/signals');

  await expect(page.getByText('观点书不可用')).toBeVisible();
  await expect(page.locator('.ledger-current')).toContainText('2026-W36');
  await expect(page.locator('.ledger-current')).not.toContainText('台账为空');
  await expect(page.locator('.ledger-history')).toContainText('1 期 · append-only');
  await expect(page.locator('.ledger-history')).not.toContainText('0 期 · append-only');
  await expect(page.locator('.ledger-track')).toContainText('已成熟至 2026-W35 · 成本 5bps');
  await expect(page.locator('.ledger-track')).not.toContainText('读取中');
  await expect(page.getByText('正在组装三尺度观点书与台账记录。')).toHaveCount(0);
});

test('generic signals current 404 is unavailable rather than an empty ledger', async ({ page }) => {
  await page.unroute('**/api/v1/signals/current');
  await page.route('**/api/v1/signals/current', async (route) => {
    await route.fulfill({ status: 404, json: { detail: 'Not Found' } });
  });

  await page.goto('/signals');

  await expect(page.getByText('短期关注风险分布。', { exact: true })).toBeVisible();
  await expect(page.getByText('本周发布暂不可用')).toBeVisible();
  await expect(page.getByText('需要第一期真实前向发布')).toHaveCount(0);
});

test('genuine typed empty ledger renders first-publication guidance', async ({ page }) => {
  await usePro(page);
  await page.goto('/signals');

  await expect(page.getByText('短期关注风险分布。', { exact: true })).toBeVisible();
  await expect(page.getByText('需要第一期真实前向发布')).toBeVisible();
  await expect(page.getByText('本周发布暂不可用')).toHaveCount(0);
  await expect(page.locator('.ledger-history')).toContainText('0 期 · append-only');
  await expect(page.locator('.ledger-history')).toContainText('尚无发布记录。');
});

for (const historyFailure of ['503', 'malformed JSON']) {
  test(`signals history ${historyFailure} remains unavailable rather than empty`, async ({ page }) => {
    await usePro(page);
    await page.unroute('**/api/v1/signals/history?*');
    await page.route('**/api/v1/signals/history?*', async (route) => {
      if (historyFailure === '503') {
        await route.fulfill({ status: 503, json: { detail: 'history store unavailable' } });
      } else {
        await route.fulfill({ status: 200, contentType: 'application/json', body: '{not-json' });
      }
    });

    await page.goto('/signals');

    await expect(page.getByText('短期关注风险分布。', { exact: true })).toBeVisible();
    await expect(page.locator('.ledger-history')).toContainText('发布历史暂不可用');
    await expect(page.locator('.ledger-history')).not.toContainText('0 期 · append-only');
    await expect(page.locator('.ledger-history')).not.toContainText('尚无发布记录。');
  });
}

test('track-record outage is unavailable rather than an empty scorecard', async ({ page }) => {
  await page.unroute('**/api/v1/signals/track-record');
  await page.route('**/api/v1/signals/track-record', async (route) => {
    await route.fulfill({ status: 503, json: { detail: 'track store unavailable' } });
  });

  await page.goto('/signals');

  await expect(page.getByText('短期关注风险分布。', { exact: true })).toBeVisible();
  await expect(page.locator('.ledger-track')).toContainText('记分服务暂不可用');
  await expect(page.locator('.ledger-track')).not.toContainText('暂无成熟计分区间（诚实空账）');
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
    await route.fulfill({ json: { total_scored: 20, directional_calls: 20, minimum_directional_calls: 20, hit_rate: 1, brier_score: 0.1, neutral_or_gated: 0, recent_outcomes: [outcome, outcome] } });
  });
  await page.goto('/quant');

  await expect(page.getByText('未观察到有效均值回归')).toBeVisible();
  await expect(page.getByText('53772 日')).toHaveCount(0);
  await expect(page.locator('.calib-item')).toHaveCount(1);
  await expect(page.getByText('估值关系显著偏离')).toBeVisible();
});

test('calibration keeps mature metrics hidden below the directional threshold', async ({ page }) => {
  await page.unroute('**/api/v1/agent/research/current');
  await page.route('**/api/v1/agent/research/current', async (route) => {
    await route.fulfill({ json: {
      data_asof: '2026-08-12', data_age_days: 1, data_stale: false, data_source: 'fixture',
      data_span: ['2004-01-01', '2026-08-12'], data_rows: 5000, degraded: {},
    } });
  });
  await page.unroute('**/api/v1/agent/calibration');
  await page.route('**/api/v1/agent/calibration', async (route) => {
    await route.fulfill({ json: {
      total_scored: 30,
      directional_calls: '19',
      minimum_directional_calls: '20',
      hit_rate: 0.84,
      brier_score: 0.12,
      neutral_or_gated: 11,
      recent_outcomes: [],
      governance: { mode: 'insufficient_data', directional_calls: '19', reason: '继续积累' },
    } });
  });
  await page.goto('/quant');

  await expect(page.getByText('正常积累中 19/20')).toBeVisible();
  await expect(page.locator('.calib-tiles')).toHaveCount(0);
  await expect(page.getByText('84%')).toHaveCount(0);
});

test('simple quant mode uses plain language instead of model abbreviations', async ({ page }) => {
  await page.goto('/quant');
  await expect(page.locator('body')).not.toContainText(/HAR-RV|HMM|BL-lite|Brier|OOS/);
});

test('simple dashboard translates data-quality machine states', async ({ page }) => {
  await page.goto('/');

  const qualityPanel = page.locator('.panel-block').filter({ hasText: '数据质量' });
  await expect(qualityPanel.getByText('部分降级').first()).toBeVisible();
  await expect(qualityPanel.getByText('部分市场指标使用代理或降级数据')).toBeVisible();
  await expect(qualityPanel).not.toContainText('market_indicators_degraded');
  await expect(qualityPanel).not.toContainText(/\bdegraded\b/i);
});

test('dashboard presents forecasts and four indicator pillars', async ({ page }) => {
  await usePro(page);
  await page.goto('/');

  await expect(page.getByRole('heading', { name: '黄金价格预测与指标总览' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '今日核心结论' })).toBeVisible();
  await expect(page.getByLabel('今日研究摘要').getByText('短期 偏多，执行倾向：观察确认')).toBeVisible();
  await expect(page.getByText('长期关注估值与配置。')).toBeVisible();
  await expect(page.getByLabel('量化模型参考').getByText('长期', { exact: true })).toBeVisible();
  await expect(page.getByText(/undefined/)).toHaveCount(0);
  await expect(page.getByText('SHORT_TERM', { exact: true })).toHaveCount(0);
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

test('source provenance badges meet small-text contrast on cream cards', async ({ page }) => {
  await usePro(page);
  await page.goto('/');

  const proxyBadge = page
    .locator('.source-health-row')
    .filter({ hasText: 'WGC Gold Demand' })
    .getByText('出处 · 代理值', { exact: true });
  await expect(proxyBadge).toBeVisible();
  expect(await computedContrast(proxyBadge)).toBeGreaterThanOrEqual(4.5);
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
  await page.getByRole('button', { name: '有帮助', exact: true }).click();
  await expect(page.getByText('已记录：这条分析有帮助。')).toBeVisible();
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

test('the long-term research category displays its valuation evidence', async ({ page }) => {
  await page.route('**/api/v1/agent/analyze', async (route) => {
    const body = route.request().postDataJSON();
    expect(body.horizon).toBe('long_term');
    await route.fulfill({ json: {
      ...analysisPayload,
      summary_card: { ...analysisPayload.summary_card, horizon: 'long_term' },
      research_focus: {
        horizon: 'long_term',
        section: { available: true, core_view: '长期估值以宏观锚为参考。', evidence: ['控制美元后的利率敏感度发生变化。'], invalidation: [] },
      },
    } });
  });
  await page.goto('/advisor');
  await advanceAdvisorMobile(page, 2);
  await page.getByRole('tab', { name: /提问分析/ }).click();
  await page.getByRole('button', { name: '长期', exact: true }).click();
  await page.getByRole('button', { name: '生成风险适配分析' }).click();

  await expect(page.getByRole('heading', { name: '长期研究观点' })).toBeVisible();
  await expect(page.getByText('长期估值以宏观锚为参考。')).toBeVisible();
  await expect(page.getByText('控制美元后的利率敏感度发生变化。')).toBeVisible();
  await expect(page.getByText('长期 · 风险适配参考')).toBeVisible();
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

test('signals page derives production ledger endpoints from the agent origin', async ({ page }) => {
  const marketView = {
    meta: {
      data_asof: '2026-08-13',
      data_age_days: 21,
      data_stale: true,
    },
    short_term: {
      available: true,
      confidence: '中',
      core_view: '短期保持观察，等待数据与价格共同确认。',
      evidence: ['波动状态偏高'],
      invalidation: ['数据质量进一步下降'],
    },
    mid_term: {
      available: true,
      confidence: '中',
      core_view: '中期由宏观因子与资金流共同决定。',
      evidence: ['美元趋势'],
      invalidation: ['实际利率快速上行'],
    },
    long_term: {
      available: true,
      confidence: '低',
      core_view: '长期估值关系处于结构性偏离期。',
      evidence: ['估值锚偏离'],
      invalidation: ['结构性需求逆转'],
    },
  };

  await page.route('https://api.goldensense.test/api/v1/agent/market-view', (route) =>
    route.fulfill({ json: marketView }),
  );
  await page.route('https://api.goldensense.test/api/v1/signals/current', (route) =>
    route.fulfill({ json: productionEmptySignalPayload }),
  );
  await page.route('https://api.goldensense.test/api/v1/signals/history?limit=26', (route) =>
    route.fulfill({ json: { publications: [] } }),
  );
  await page.route('https://api.goldensense.test/api/v1/signals/track-record', (route) =>
    route.fulfill({ status: 503, json: { detail: { message: '暂无成熟周' } } }),
  );
  await page.route('https://api.goldensense.test/api/v1/agent/research/current', (route) =>
    route.fulfill({ json: { cross_asset: null } }),
  );

  // Reproduce the SPA fallback when ledger requests use the web app's origin.
  await page.route('http://127.0.0.1:4173/api/v1/signals/**', (route) =>
    route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>GoldenSense</title>' }),
  );

  await page.goto('/signals');

  await expect(page.getByText('短期保持观察，等待数据与价格共同确认。')).toBeVisible();
  await expect(page.getByText("Unexpected token '<'")).toHaveCount(0);
  await expect(page.getByRole('heading', { name: '本周发布' })).toBeVisible();
  await expect(page.getByText('台账为空', { exact: true })).toBeVisible();
});

test('quant page explains a slow first-load model warm-up', async ({ page }) => {
  await page.route('https://api.goldensense.test/api/v1/agent/research/current', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 7_000));
    await route.fulfill({ json: { data_span: ['2004-01-05', '2026-08-13'], data_rows: 5706 } });
  });
  await page.route('https://api.goldensense.test/api/v1/agent/calibration', (route) =>
    route.fulfill({ status: 503, json: { detail: { message: '样本不足' } } }),
  );

  await page.goto('/quant');

  await expect(page.getByRole('status').filter({ hasText: '首次打开正在唤醒研究服务' })).toBeVisible({
    timeout: 6_500,
  });
});

test('quant page displays driver attribution and matured valuation evidence', async ({ page }) => {
  await page.route('**/api/v1/agent/research/current', (route) => route.fulfill({
    json: {
      macro_factors: {
        latest: { usd_downtrend: 1 },
        factors_used: ['usd_downtrend'],
        composite: 1,
      },
      degraded: { macro_vintages: 'unavailable:2Y_CMT' },
      gold_drivers: {
        data_asof: '2026-07-14',
        flows: { etf_net_tonnes: { label: '黄金 ETF 净流入', status: 'unavailable' } },
      },
      driver_attribution: { data_asof: '2026-07-14', latest: { actual: 0.01, dollar: 0.004, residual: 0.006 } },
      vol_bands: { h1: { p10: -0.01, p50: 0, p90: 0.01, ann_vol_forecast: 0.2 } },
      risk_review: {
        price_end: '2026-07-14',
        windows: {
          '1': {
            log_har: { n: 82, coverage: 0.817, status: 'hold', blockers: ['session_timestamps_missing'] },
            student_t_garch: { n: null, coverage: null, status: 'hold', blockers: ['incomplete_forecasts'] },
          },
        },
      },
      fair_value: {
        deviation_pct: 1.0,
        anchor_status: 'conditional',
        half_life_days: 40,
        quartile_forward_returns: { q1_cheapest: 0.08 },
        quartile_sample_counts: { q1_cheapest: 72 },
        stress_scenarios_pct: { real_yield_up_100bp: -2.2, dollar_up_5pct: -1.1, joint_up: -3.25 },
      },
    },
  }));
  await page.route('**/api/v1/agent/calibration', (route) => route.fulfill({ json: {} }));
  await usePro(page);
  await page.goto('/quant');
  await expect(page.getByText('部分宏观历史版本缺失', { exact: false })).toBeVisible();
  await expect(page.getByRole('heading', { name: '当期驱动归因' })).toBeVisible();
  await expect(page.getByText('72 个成熟样本')).toBeVisible();
  await expect(page.getByRole('heading', { name: '利率与美元条件锚' })).toBeVisible();
  await expect(page.getByText('未观察到有效均值回归')).toBeVisible();
  await expect(page.getByText('40 日', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('heading', { name: '长期估值压力情景' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '黄金驱动证据' })).toBeVisible();
  await expect(page.getByText('缺少已发布数据')).toBeVisible();
  await expect(page.getByRole('heading', { name: '短期风险候选复验' })).toBeVisible();
  await expect(page.getByText('82 个成熟样本 · 暂缓晋级')).toBeVisible();
  await expect(page.getByText('缺少核验会话')).toBeVisible();
  await expect(page.getByText('Student-t GARCH', { exact: true })).toBeVisible();
  await expect(page.getByText('成熟样本数待确认 · 暂缓晋级')).toBeVisible();
  await expect(page.getByText('候选模型的预测记录不完整')).toBeVisible();
});

test('signals page labels date-only proxy scoring and same-cost benchmarks', async ({ page }) => {
  await page.route('**/api/v1/agent/market-view', (route) =>
    route.fulfill({ json: { meta: {} } }),
  );
  await page.route('**/api/v1/signals/current', (route) =>
    route.fulfill({ json: productionEmptySignalPayload }),
  );
  await page.route('**/api/v1/signals/history?*', (route) =>
    route.fulfill({ json: { publications: [] } }),
  );
  await page.route('**/api/v1/signals/track-record', (route) =>
    route.fulfill({
      json: {
        price_evidence: 'date_only_proxy',
        per_profile: {
          conservative: { weeks_scored: 2, cum_return: 0.02 },
          balanced: {
            intervals_scored: 1, weeks_scored: 2, cum_return: 0.04, max_drawdown: -0.01,
            ann_vol: null, sharpe: null, annualization_reason: 'nonweekly_intervals',
          },
        },
        benchmarks: {
          conservative_risk_matched_net: { weeks_scored: 0, cum_return: null },
          balanced_fixed_net: { cum_return: 0.03 },
          balanced_risk_matched_net: {
            weeks_scored: 1,
            cum_return: 0.025,
            target_ann_vol: 0.08,
            start_at: '2026-07-20T18:00:00+00:00',
            strategy_same_period: { weeks_scored: 1, cum_return: 0.035 },
          },
        },
      },
    }),
  );
  await page.route('**/api/v1/agent/research/current', (route) =>
    route.fulfill({ json: {} }),
  );
  await page.goto('/signals');
  await expect(page.getByText('行情按日期估算可用时间，记分供研究参考。')).toBeVisible();
  await expect(page.getByText('同成本固定仓位 3.00%')).toBeVisible();
  await expect(page.locator('.track-row').filter({ hasText: '稳健' })).toContainText('区间数 1');
  await expect(page.locator('.track-row').filter({ hasText: '保守' })).toContainText('区间数 2');
  await expect(page.getByText('计分区间包含非周度间隔，年化波动率与 Sharpe 暂不可用。')).toBeVisible();
  await expect(page.getByText('同成本风险匹配基准 2.50% · 同期策略 3.50% · 目标年化波动率 8.0% · 自 2026-07-20 起')).toBeVisible();
  await expect(page.getByText(/同成本风险匹配基准/)).toHaveCount(1);
});

for (const scenario of [
  {
    name: 'suitability disallows a calculable status',
    suitability: { status: 'insufficient', position_analysis_allowed: false, missing_fields: [], reasons: [] },
    positionGap: { status: 'within', current_gold_pct: 60, gap_pct: 53.45 },
    explanation: '适当性校验未允许个人仓位分析',
  },
  {
    name: 'the position status is unrecognized',
    suitability: { status: 'eligible', position_analysis_allowed: true, missing_fields: [], reasons: [] },
    positionGap: { status: 'future_status', current_gold_pct: 60, gap_pct: 53.45 },
    explanation: '仓位差距状态不可识别',
  },
]) {
  test(`advisor fails closed when ${scenario.name}`, async ({ page }) => {
    await page.unroute('**/api/v1/agent/personal-research?*');
    await page.route('**/api/v1/agent/personal-research?*', async (route) => {
      await route.fulfill({ json: {
        ...personalResearchPayload,
        facts: {
          ...personalResearchPayload.facts,
          reference_range: { available: true, range_pct: [4, 9.1], midpoint: 6.55 },
          suitability: scenario.suitability,
          position_gap: scenario.positionGap,
        },
        narrative: {
          ...personalResearchPayload.narrative,
          position_analysis: '敏感的个人仓位判断不应展示',
        },
      } });
    });
    await page.goto('/advisor');
    await advanceAdvisorMobile(page, 2);
    await page.getByRole('button', { name: /生成个性化配置研究/ }).click();

    const rangePanel = page.locator('.advisor-range-panel');
    await expect(rangePanel).toContainText('4.0% – 9.1%');
    await expect(rangePanel).toContainText('当前仓位 60%');
    await expect(rangePanel).toContainText('个人仓位差距暂不计算');
    await expect(rangePanel).toContainText(scenario.explanation);
    await expect(rangePanel.getByText('处于参考区间内', { exact: true })).toHaveCount(0);
    await expect(page.getByText('敏感的个人仓位判断不应展示', { exact: true })).toHaveCount(0);
  });
}
