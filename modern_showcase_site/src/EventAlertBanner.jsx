import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Siren, X } from 'lucide-react';

const API_URL = import.meta.env.VITE_AGENT_API_URL || '/api/v1/agent/analyze';
const ALERT_URL =
  import.meta.env.VITE_AGENT_EVENT_ALERT_URL || API_URL.replace('/analyze', '/event-alert');
const API_KEY = import.meta.env.VITE_AGENT_API_KEY || 'dev-public-key';

const categoryLabels = {
  monetary_policy: '货币政策',
  inflation: '通胀',
  geopolitics: '地缘',
  usd: '美元',
  flows: '资金流',
};

// Site-wide high-severity event banner. Fed by /event-alert (5-minute
// server-side cache); dismissal is remembered per event per session.
export default function EventAlertBanner() {
  const [alert, setAlert] = useState(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetch(ALERT_URL, { headers: { 'X-API-Key': API_KEY }, signal: controller.signal })
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        if (!payload?.active) return;
        const key = `gs_alert_dismissed_${payload.title}`;
        if (window.sessionStorage.getItem(key)) return;
        setAlert(payload);
      })
      .catch(() => {
        /* alert feed unavailable -> no banner, never an error */
      });
    return () => controller.abort();
  }, []);

  if (!alert || dismissed) return null;

  function dismiss() {
    try {
      window.sessionStorage.setItem(`gs_alert_dismissed_${alert.title}`, '1');
    } catch {
      /* session-only */
    }
    setDismissed(true);
  }

  return (
    <div className="event-alert-banner" role="alert">
      <span className="alert-pulse" aria-hidden="true" />
      <Siren size={15} />
      <strong>重大市场事件{alert.category ? `（${categoryLabels[alert.category] || alert.category}）` : ''}</strong>
      <span className="alert-title">{alert.title}</span>
      <Link to="/signals">观点书将在下次刷新纳入影响 →</Link>
      <button type="button" aria-label="关闭事件提醒" onClick={dismiss}>
        <X size={14} />
      </button>
    </div>
  );
}
