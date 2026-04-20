from __future__ import annotations

from datetime import timezone
from types import SimpleNamespace

import pytest

from frontend.vix_data import VixDataError, fetch_vix_latest, validate_vix_value


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http_error")

    def json(self):
        return self._payload


def test_validate_vix_value_range_ok():
    assert validate_vix_value(18.3) == 18.3


def test_validate_vix_value_out_of_range():
    with pytest.raises(VixDataError):
        validate_vix_value(-1.0)
    with pytest.raises(VixDataError):
        validate_vix_value(1000.0)


def test_fetch_vix_latest_parses_yahoo(monkeypatch):
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1, 2, 3],
                    "indicators": {"quote": [{"close": [None, 16.0, 17.5]}]},
                }
            ],
            "error": None,
        }
    }

    def _fake_get(url, timeout, headers):
        return _Resp(payload, 200)

    import frontend.vix_data as m

    monkeypatch.setattr(m.requests, "get", _fake_get)
    snap = fetch_vix_latest(timeout_s=1.0)
    assert snap.value == 17.5
    assert snap.timestamp.tzinfo == timezone.utc
    assert snap.source == "YahooFinance"


def test_fetch_vix_latest_network_error(monkeypatch):
    def _fake_get(url, timeout, headers):
        raise RuntimeError("network_down")

    import frontend.vix_data as m

    monkeypatch.setattr(m.requests, "get", _fake_get)
    with pytest.raises(VixDataError):
        fetch_vix_latest(timeout_s=1.0)


def test_fetch_vix_latest_bad_payload(monkeypatch):
    payload = {"chart": {"result": [], "error": None}}

    def _fake_get(url, timeout, headers):
        return _Resp(payload, 200)

    import frontend.vix_data as m

    monkeypatch.setattr(m.requests, "get", _fake_get)
    with pytest.raises(VixDataError):
        fetch_vix_latest(timeout_s=1.0)
