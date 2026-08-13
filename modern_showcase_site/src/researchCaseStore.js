import { useEffect, useState } from 'react';

const STORAGE_KEY = 'gs_active_research_case_v1';
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
    evidence_documents: researchCase.evidence_documents || [],
    fact_claims: researchCase.fact_claims || [],
    gate_report: researchCase.gate_report || [],
    model_registry: researchCase.model_registry || [],
    agent_views: researchCase.agent_views || [],
    conflicts: researchCase.conflicts || [],
    horizon_strategy: researchCase.horizon_strategy || {},
    audit_report: researchCase.audit_report || null,
    personalized_brief: researchCase.personalized_brief || null,
  };
}

export function loadActiveCase() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function saveActiveCase(researchCase) {
  const compact = compactCase(researchCase);
  try {
    if (compact) window.localStorage.setItem(STORAGE_KEY, JSON.stringify(compact));
    else window.localStorage.removeItem(STORAGE_KEY);
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
