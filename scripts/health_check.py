"""Operational health check: the operator's eyes, cron-ready.

Checks the running stack and exits non-zero when anything needs attention,
so a plain crontab line (or CI schedule) becomes an alerting system:

    */30 * * * * cd /path/to/repo && python3 scripts/health_check.py || true

Checks:
  1. gateway /health reachable and status ok
  2. research data freshness (data_age_days within threshold)
  3. /metrics 5xx error rate over threshold
  4. this ISO week's signal publication exists (Mon 12:00 UTC grace)

When ALERT_WEBHOOK_URL is set, failures are POSTed as JSON (Slack-compatible
``text`` field included). Webhook delivery failures are logged, never raised
-- the exit code is the source of truth.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_loader import load_env_file  # noqa: E402

load_env_file()

LOGGER = logging.getLogger("health_check")

GATEWAY_BASE = os.environ.get("HEALTH_GATEWAY_BASE", "http://127.0.0.1:8020")
DATA_AGE_MAX_DAYS = int(os.environ.get("HEALTH_DATA_AGE_MAX_DAYS", "4"))
ERROR_RATE_MAX = float(os.environ.get("HEALTH_5XX_RATE_MAX", "0.05"))
PUBLICATION_GRACE_HOUR_UTC = 12  # Monday noon UTC before "missing" fires


def _default_fetch(url: str, *, timeout: float = 6.0) -> Dict[str, Any]:
    # /metrics is an internal-only endpoint; everything else is public.
    key_env = "AGENT_INTERNAL_API_KEYS" if url.endswith("/metrics") else "AGENT_PUBLIC_API_KEYS"
    fallback = "dev-internal-key" if url.endswith("/metrics") else "dev-public-key"
    request = urllib.request.Request(url, headers={
        "X-API-Key": os.environ.get(key_env, fallback).split(",")[0],
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_checks(
    *,
    fetch: Optional[Callable[[str], Dict[str, Any]]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    fetch = fetch or _default_fetch
    now = now or datetime.now(timezone.utc)
    failures: List[str] = []
    checks: Dict[str, Any] = {}

    # 1. gateway health
    try:
        health = fetch(f"{GATEWAY_BASE}/health")
        checks["gateway"] = health.get("status")
        if health.get("status") != "ok":
            failures.append(f"gateway_status={health.get('status')}")
    except Exception as exc:
        checks["gateway"] = "unreachable"
        failures.append(f"gateway_unreachable: {type(exc).__name__}")

    # 2. data freshness
    try:
        research = fetch(f"{GATEWAY_BASE}/api/v1/agent/research/current")
        age = research.get("data_age_days")
        checks["data_age_days"] = age
        if age is None or age > DATA_AGE_MAX_DAYS:
            failures.append(f"data_stale: age_days={age} > {DATA_AGE_MAX_DAYS}")
    except Exception as exc:
        checks["data_age_days"] = None
        failures.append(f"research_unreachable: {type(exc).__name__}")

    # 3. 5xx error rate (top-level status_classes from service_metrics)
    try:
        metrics = fetch(f"{GATEWAY_BASE}/metrics")
        classes = metrics.get("status_classes") or {}
        total = sum(int(v) for v in classes.values())
        errors = int(classes.get("5xx", 0))
        rate = (errors / total) if total else 0.0
        checks["error_rate_5xx"] = round(rate, 4)
        if rate > ERROR_RATE_MAX:
            failures.append(f"error_rate_5xx={rate:.2%} > {ERROR_RATE_MAX:.0%}")
    except Exception as exc:
        checks["error_rate_5xx"] = None
        failures.append(f"metrics_unreachable: {type(exc).__name__}")

    # 4. weekly publication exists (after Monday-noon grace)
    iso = now.isocalendar()
    expected_id = f"{iso[0]}-W{iso[1]:02d}"
    past_grace = now.weekday() > 0 or now.hour >= PUBLICATION_GRACE_HOUR_UTC
    try:
        current = fetch(f"{GATEWAY_BASE}/api/v1/signals/current")
        checks["latest_publication"] = current.get("publication_id")
        if past_grace and current.get("publication_id") != expected_id:
            failures.append(
                f"publication_missing: expected {expected_id}, "
                f"latest {current.get('publication_id')}"
            )
    except Exception as exc:
        checks["latest_publication"] = None
        if past_grace:
            failures.append(f"publication_missing: {type(exc).__name__}")

    return {
        "ok": not failures,
        "checked_at": now.isoformat(),
        "checks": checks,
        "failures": failures,
    }


def post_webhook(report: Dict[str, Any]) -> bool:
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url or report["ok"]:
        return False
    payload = {
        "text": "GoldenSense 健康检查失败: " + "; ".join(report["failures"]),
        "report": report,
    }
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=5)
        return True
    except Exception as exc:
        LOGGER.warning("alert_webhook_failed error=%s:%s", type(exc).__name__, exc)
        return False


def main(argv: Optional[list] = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print full JSON report")
    args = parser.parse_args(argv)

    report = run_checks()
    report["webhook_sent"] = post_webhook(report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        status = "OK" if report["ok"] else "FAIL"
        print(f"health_check {status} checks={report['checks']} "
              f"failures={report['failures']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
