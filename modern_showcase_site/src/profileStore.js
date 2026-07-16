// Unified investor profile store, shared by /advisor and /agent.
// One localStorage key, core 4 fields + optional advanced layer.
// Migrates the old advisor v1 profile transparently.

const STORAGE_KEY = 'gs_profile_v2';
const LEGACY_ADVISOR_KEY = 'gs_advisor_profile_v1';

export const defaultProfile = {
  // Core layer (always present)
  risk_tolerance: 'balanced',
  horizon: 'mid',
  current_gold_pct: 10,
  experience: 'novice',
  // Advanced layer (null = not provided; backend rules stay silent)
  max_drawdown_pct: null,
  liquidity_need: null,
  leverage_attitude: null,
  investment_goal: null,
};

export function loadProfile() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...defaultProfile, ...JSON.parse(raw) };
    // One-time migration from the advisor v1 store.
    const legacy = window.localStorage.getItem(LEGACY_ADVISOR_KEY);
    if (legacy) {
      const migrated = { ...defaultProfile, ...JSON.parse(legacy) };
      saveProfile(migrated);
      return migrated;
    }
  } catch {
    /* corrupted storage -> defaults */
  }
  return { ...defaultProfile };
}

export function saveProfile(profile) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(profile));
  } catch {
    /* storage unavailable: session-only */
  }
}

// Request body for POST /personal-research: numbers coerced, nulls dropped.
export function toPersonalResearchBody(profile) {
  const body = {
    risk_tolerance: profile.risk_tolerance,
    horizon: profile.horizon,
    current_gold_pct: Number(profile.current_gold_pct),
    experience: profile.experience,
  };
  if (profile.max_drawdown_pct !== null && profile.max_drawdown_pct !== '') {
    body.max_drawdown_pct = Number(profile.max_drawdown_pct);
  }
  for (const key of ['liquidity_need', 'leverage_attitude', 'investment_goal']) {
    if (profile[key]) body[key] = profile[key];
  }
  return body;
}

// Adapter to the legacy /analyze investor_profile contract (all 9 fields
// required there). The unified profile is the source of truth; missing
// advanced fields fall back to the contract's most conservative defaults.
const RISK_TO_CAPACITY = { conservative: 'low', balanced: 'medium', aggressive: 'high' };
const HORIZON_TO_TRADING = { short: 'short', mid: 'medium', long: 'long' };
const EXPERIENCE_TO_LEVEL = {
  novice: 'beginner',
  experienced: 'intermediate',
  professional: 'advanced',
};

export function toLegacyAnalyzeProfile(profile) {
  const gold = Number(profile.current_gold_pct) || 0;
  return {
    risk_capacity: RISK_TO_CAPACITY[profile.risk_tolerance] || 'medium',
    trading_horizon: HORIZON_TO_TRADING[profile.horizon] || 'medium',
    experience_level: EXPERIENCE_TO_LEVEL[profile.experience] || 'intermediate',
    capital_allocation_pct: gold,
    max_drawdown_pct:
      profile.max_drawdown_pct !== null && profile.max_drawdown_pct !== ''
        ? Number(profile.max_drawdown_pct)
        : 10,
    current_position: gold > 0 ? 'long' : 'none',
    liquidity_need: profile.liquidity_need || 'medium',
    leverage_attitude: profile.leverage_attitude || 'none',
    investment_goal: profile.investment_goal || 'trend_following',
  };
}
