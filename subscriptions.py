"""Weekly-digest subscriptions: the product's lightest retention loop.

No account system. An email address plus an unsubscribe token, stored in a
local JSONL file (``data_cache/`` is gitignored -- personal data never enters
the repo). Every weekly signal publication can be rendered into a plain-text
digest and delivered through a pluggable transport:

- SMTP when ``SMTP_HOST`` is configured (industrial default),
- an injected callable in tests,
- otherwise an honest ``log_only`` degradation -- the report says the mail
  was NOT delivered rather than pretending it was.

Every digest body carries the recipient's personal unsubscribe link and the
publication's tamper-evident content hash.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger("subscriptions")

DEFAULT_STORE_PATH = "data_cache/subscriptions.jsonl"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")

PROFILE_ZH = {"conservative": "保守", "balanced": "稳健", "aggressive": "进取"}
HORIZON_ZH = {"short_term": "短期", "mid_term": "中期", "long_term": "长期"}


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


class SubscriptionStore:
    """JSONL-backed store; the newest record per email wins."""

    def __init__(self, path: str | Path = DEFAULT_STORE_PATH):
        self._path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        state: Dict[str, Dict[str, Any]] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    record = json.loads(line)
                    state[record["email"]] = record
        return state

    def _append(self, record: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    def subscribe(self, email: str) -> Dict[str, Any]:
        normalized = (email or "").strip().lower()
        if not _EMAIL_RE.match(normalized):
            raise ValueError(f"invalid email address: {email!r}")
        with self._lock:
            state = self._load()
            existing = state.get(normalized)
            if existing and existing.get("active"):
                return {"email": normalized, "token": existing["token"],
                        "created": False}
            record = {
                "email": normalized,
                "token": secrets.token_urlsafe(24),
                "active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._append(record)
            return {"email": normalized, "token": record["token"], "created": True}

    def unsubscribe(self, token: str) -> bool:
        if not token:
            return False
        with self._lock:
            state = self._load()
            for record in state.values():
                if record.get("token") == token and record.get("active"):
                    self._append({**record, "active": False,
                                  "unsubscribed_at": datetime.now(timezone.utc).isoformat()})
                    return True
        return False

    def active_subscribers(self) -> List[Dict[str, Any]]:
        return [r for r in self._load().values() if r.get("active")]


# --------------------------------------------------------------------------- #
# Digest rendering
# --------------------------------------------------------------------------- #
def render_digest(
    publication: Dict[str, Any],
    track_record: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Plain-text weekly digest. ``{unsubscribe_url}`` is a per-recipient
    placeholder filled at send time."""
    pub_id = publication.get("publication_id", "?")
    lines: List[str] = [
        f"GoldenSense 每周信号 · {pub_id}",
        f"数据截至 {publication.get('data_asof')}（EOD，非实时）",
        "",
        "本周研究参考区间（组合占比）：",
    ]
    for profile, alloc in (publication.get("allocations") or {}).items():
        rng = alloc.get("range_pct")
        if rng:
            lines.append(
                f"  {PROFILE_ZH.get(profile, profile)}: "
                f"{rng[0]:.1f}%–{rng[1]:.1f}%（中点 {alloc.get('midpoint')}%）"
            )
    lines.append("")
    lines.append("观点书摘要：")
    for horizon, summary in (publication.get("market_view_summary") or {}).items():
        if summary:
            lines.append(f"  [{HORIZON_ZH.get(horizon, horizon)}] {summary}")

    if track_record and track_record.get("matured_through"):
        lines.append("")
        lines.append(
            f"前向记分卡（已成熟至 {track_record['matured_through']}，"
            f"含 {track_record.get('cost_bps')}bps 换手成本）："
        )
        for profile, stats in (track_record.get("per_profile") or {}).items():
            if stats.get("weeks_scored"):
                cum = stats.get("cum_return")
                lines.append(
                    f"  {PROFILE_ZH.get(profile, profile)}: {stats['weeks_scored']} 周 · "
                    f"累计 {cum * 100:.2f}%" if cum is not None else ""
                )
        bench = ((track_record.get("benchmarks") or {}).get("gold_buy_hold") or {})
        if bench.get("cum_return") is not None:
            lines.append(f"  基准（黄金买入持有）: {bench['cum_return'] * 100:.2f}%")

    lines += [
        "",
        f"本期记录不可变哈希：{publication.get('content_hash', '')[:24]}…",
        "台账 append-only、前向计分、不回填历史。",
        "",
        publication.get("disclaimer", ""),
        "",
        "退订：{unsubscribe_url}",
    ]
    return {
        "subject": f"GoldenSense 每周信号 {pub_id}",
        "text": "\n".join(lines),
    }


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #
def _smtp_transport() -> Optional[Callable[[str, str, str], None]]:
    host = os.environ.get("SMTP_HOST")
    if not host:
        return None
    import smtplib
    from email.mime.text import MIMEText

    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "")
    password = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("SMTP_FROM", user or "goldensense@localhost")

    def transport(to_addr: str, subject: str, body: str) -> None:
        message = MIMEText(body, "plain", "utf-8")
        message["Subject"] = subject
        message["From"] = sender
        message["To"] = to_addr
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.ehlo()
            if os.environ.get("SMTP_STARTTLS", "1") == "1":
                server.starttls()
            if user:
                server.login(user, password)
            server.sendmail(sender, [to_addr], message.as_string())

    return transport


def send_digest(
    publication: Dict[str, Any],
    track_record: Optional[Dict[str, Any]],
    *,
    store: Optional[SubscriptionStore] = None,
    base_url: str = "",
    transport: Optional[Callable[[str, str, str], None]] = None,
) -> Dict[str, Any]:
    """Send the weekly digest to every active subscriber.

    Returns an honest delivery report; a missing SMTP config is reported as
    ``log_only`` with zero delivered, never silently swallowed.
    """
    store = store or SubscriptionStore()
    subscribers = store.active_subscribers()
    digest = render_digest(publication, track_record)

    report: Dict[str, Any] = {
        "publication_id": publication.get("publication_id"),
        "recipients": len(subscribers),
        "delivered": 0,
        "failed": 0,
        "transport": "injected" if transport else "smtp",
        "reason": "",
    }
    if transport is None:
        transport = _smtp_transport()
        if transport is None:
            report["transport"] = "log_only"
            report["reason"] = "smtp_not_configured; digest logged, not delivered"
            LOGGER.info(
                "digest_log_only publication=%s recipients=%d subject=%r",
                report["publication_id"], len(subscribers), digest["subject"],
            )
            return report

    base = (base_url or os.environ.get("PUBLIC_BASE_URL", "")).rstrip("/")
    for subscriber in subscribers:
        unsubscribe_url = (
            f"{base}/api/v1/subscriptions/unsubscribe?token={subscriber['token']}"
        )
        body = digest["text"].replace("{unsubscribe_url}", unsubscribe_url)
        try:
            transport(subscriber["email"], digest["subject"], body)
            report["delivered"] += 1
        except Exception as exc:
            report["failed"] += 1
            LOGGER.warning(
                "digest_send_failed to=%s error=%s:%s",
                mask_email(subscriber["email"]), type(exc).__name__, exc,
            )
    return report
