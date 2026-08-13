import { useEffect, useState } from 'react';

const STORAGE_KEY = 'gs_active_research_case_v1';
const SESSION_CASE_KEY = 'gs_active_research_case_session_v1';
const SESSION_OWNER_KEY = 'gs_research_session_v1';
const EVENT_NAME = 'goldensense:research-case';

function compactCase(researchCase) {
  if (!researchCase) return null;
  return {
    case_id: researchCase.case_id,
    question: researchCase.question,
    asset: researchCase.asset,
    created_at: researchCase.created_at,
    data_asof: researchCase.data_asof,
    status: researchCase.status,
    research_mode: researchCase.research_mode,
    evidence_documents: researchCase.evidence_documents || [],
    fact_claims: researchCase.fact_claims || [],
    gate_report: researchCase.gate_report || [],
    model_registry: researchCase.model_registry || [],
    agent_views: researchCase.agent_views || [],
    conflicts: researchCase.conflicts || [],
    horizon_strategy: researchCase.horizon_strategy || {},
    audit_report: researchCase.audit_report || null,
    personalized_brief: researchCase.personalized_brief || null,
    narrative: researchCase.narrative || null,
    outcome_schedule: researchCase.outcome_schedule || [],
  };
}

export function getResearchSession() {
  let token = window.sessionStorage.getItem(SESSION_OWNER_KEY);
  if (!token) {
    token = `gs-${window.crypto.randomUUID()}`;
    window.sessionStorage.setItem(SESSION_OWNER_KEY, token);
  }
  return token;
}

export function getResearchHeaders(apiKey) {
  return {
    'X-API-Key': apiKey,
    'X-Research-Session': getResearchSession(),
  };
}

export function loadActiveCase() {
  try {
    const raw = window.sessionStorage.getItem(SESSION_CASE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function saveActiveCase(researchCase) {
  const compact = compactCase(researchCase);
  try {
    if (compact) {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ case_id: compact.case_id }));
      window.sessionStorage.setItem(SESSION_CASE_KEY, JSON.stringify(compact));
    } else {
      window.localStorage.removeItem(STORAGE_KEY);
      window.sessionStorage.removeItem(SESSION_CASE_KEY);
    }
  } catch {
    // Storage-disabled browsers still keep the case in the current component.
  }
  window.dispatchEvent(new CustomEvent(EVENT_NAME, { detail: compact }));
  return compact;
}

export function useActiveResearchCase() {
  const [activeCase, setActiveCase] = useState(loadActiveCase);

  useEffect(() => {
    const onCase = (event) => setActiveCase(event.detail || loadActiveCase());
    const onStorage = (event) => {
      if (event.key === STORAGE_KEY) setActiveCase(loadActiveCase());
    };
    window.addEventListener(EVENT_NAME, onCase);
    window.addEventListener('storage', onStorage);
    return () => {
      window.removeEventListener(EVENT_NAME, onCase);
      window.removeEventListener('storage', onStorage);
    };
  }, []);

  return activeCase;
}
