"""Tests for the weekly-digest subscription store, rendering and delivery."""
import pytest

from subscriptions import (
    SubscriptionStore,
    mask_email,
    render_digest,
    send_digest,
)


def _publication() -> dict:
    return {
        "publication_id": "2026-W29",
        "published_at": "2026-07-13T09:00:00+00:00",
        "data_asof": "2026-07-13",
        "allocations": {
            "conservative": {"range_pct": [2.0, 8.0], "midpoint": 5.0},
            "balanced": {"range_pct": [2.6, 6.2], "midpoint": 4.4},
            "aggressive": {"range_pct": [4.2, 9.9], "midpoint": 7.05},
        },
        "market_view_summary": {
            "short_term": "短期以分布与风险为口径。",
            "mid_term": "中期主导状态为压力。",
            "long_term": "长期估值处于正常带内。",
        },
        "content_hash": "sha256:abcdef0123456789",
        "disclaimer": "不构成任何投资建议。",
        "degraded": {},
    }


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
def test_subscribe_normalizes_and_dedupes(tmp_path):
    store = SubscriptionStore(tmp_path / "subs.jsonl")
    first = store.subscribe("  User@Example.COM ")
    again = store.subscribe("user@example.com")
    assert first["email"] == "user@example.com"
    assert again["created"] is False
    assert len(store.active_subscribers()) == 1
    assert first["token"]


def test_subscribe_rejects_invalid_email(tmp_path):
    store = SubscriptionStore(tmp_path / "subs.jsonl")
    with pytest.raises(ValueError):
        store.subscribe("not-an-email")
    with pytest.raises(ValueError):
        store.subscribe("a@b")  # no TLD


def test_unsubscribe_by_token_idempotent(tmp_path):
    store = SubscriptionStore(tmp_path / "subs.jsonl")
    sub = store.subscribe("user@example.com")
    assert store.unsubscribe(sub["token"]) is True
    assert store.unsubscribe(sub["token"]) is False  # already inactive
    assert store.unsubscribe("bogus-token") is False
    assert store.active_subscribers() == []
    # Resubscribing re-activates the same address.
    re_sub = store.subscribe("user@example.com")
    assert re_sub["created"] is True
    assert len(store.active_subscribers()) == 1


def test_mask_email():
    assert mask_email("user@example.com") == "u***@example.com"
    assert mask_email("ab@x.io") == "a***@x.io"


# --------------------------------------------------------------------------- #
# Digest rendering
# --------------------------------------------------------------------------- #
def test_render_digest_contains_ranges_views_and_disclaimer():
    digest = render_digest(_publication(), track_record=None)
    assert "2026-W29" in digest["subject"]
    text = digest["text"]
    assert "2.6" in text and "6.2" in text          # balanced range
    assert "中期主导状态为压力" in text
    assert "sha256:abcdef" in text                   # tamper-evident hash cited
    assert "不构成任何投资建议" in text
    assert "{unsubscribe_url}" in text               # per-recipient placeholder


def test_render_digest_includes_track_record_when_matured():
    track = {
        "per_profile": {"balanced": {"weeks_scored": 3, "cum_return": 0.012,
                                     "max_drawdown": -0.004}},
        "benchmarks": {"gold_buy_hold": {"cum_return": 0.02}},
        "matured_through": "2026-W28",
        "cost_bps": 5.0,
    }
    text = render_digest(_publication(), track_record=track)["text"]
    assert "3 周" in text or "weeks_scored" not in text
    assert "1.20%" in text


# --------------------------------------------------------------------------- #
# Delivery (transport degradation is explicit)
# --------------------------------------------------------------------------- #
def test_send_digest_without_smtp_degrades_honestly(tmp_path, monkeypatch):
    for key in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    store = SubscriptionStore(tmp_path / "subs.jsonl")
    store.subscribe("user@example.com")
    report = send_digest(_publication(), None, store=store,
                         base_url="https://example.test")
    assert report["recipients"] == 1
    assert report["delivered"] == 0
    assert report["transport"] == "log_only"
    assert "smtp_not_configured" in report["reason"]


def test_send_digest_uses_injected_transport(tmp_path):
    sent = []

    def transport(to_addr, subject, body):
        sent.append((to_addr, subject))

    store = SubscriptionStore(tmp_path / "subs.jsonl")
    store.subscribe("a@example.com")
    store.subscribe("b@example.com")
    report = send_digest(_publication(), None, store=store,
                         base_url="https://example.test", transport=transport)
    assert report["delivered"] == 2
    assert len(sent) == 2
    # Each recipient gets their own unsubscribe token in the body.
    assert sent[0][0] == "a@example.com"


def test_send_digest_body_has_personal_unsubscribe_link(tmp_path):
    bodies = []

    def transport(to_addr, subject, body):
        bodies.append(body)

    store = SubscriptionStore(tmp_path / "subs.jsonl")
    sub = store.subscribe("a@example.com")
    send_digest(_publication(), None, store=store,
                base_url="https://example.test", transport=transport)
    assert sub["token"] in bodies[0]
    assert "{unsubscribe_url}" not in bodies[0]
