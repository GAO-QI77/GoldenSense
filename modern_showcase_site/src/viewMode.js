// Global view mode: '简明' (simple, default) vs '专业' (pro).
// Simple mode is the C-end door: it hides methodology internals (indicator
// audits, source health, citation chains, ledger hashes) behind one toggle,
// without ever hiding risk-relevant content (invalidation conditions,
// disclaimers and freshness banners stay in both modes).
import { createContext, useContext } from 'react';

const STORAGE_KEY = 'gs_view_mode';

export const ViewModeContext = createContext({ mode: 'simple', setMode: () => {} });

export function useViewMode() {
  return useContext(ViewModeContext);
}

export function loadViewMode() {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored === 'pro' ? 'pro' : 'simple';
  } catch {
    return 'simple';
  }
}

export function saveViewMode(mode) {
  try {
    window.localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    /* session-only */
  }
}
