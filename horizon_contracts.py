"""Public research-horizon contract and isolated legacy model adapter.

Product and API consumers use research periods.  The old T+ model keys are
kept behind these functions so inference compatibility cannot leak back into
the user-facing contract.
"""
from __future__ import annotations

from typing import Literal, cast


PublicHorizon = Literal["short_term", "mid_term", "long_term"]
LegacyQuantHorizon = Literal["T+1", "T+7", "T+30"]

PUBLIC_HORIZONS: tuple[PublicHorizon, ...] = (
    "short_term",
    "mid_term",
    "long_term",
)

HORIZON_LABELS: dict[PublicHorizon, str] = {
    "short_term": "短期",
    "mid_term": "中期",
    "long_term": "长期",
}

HORIZON_WINDOWS: dict[PublicHorizon, str] = {
    "short_term": "1–21天",
    "mid_term": "1–6月",
    "long_term": "6月以上",
}

_PUBLIC_TO_LEGACY_QUANT: dict[PublicHorizon, LegacyQuantHorizon] = {
    "short_term": "T+1",
    "mid_term": "T+7",
    "long_term": "T+30",
}

_LEGACY_TO_PUBLIC: dict[str, PublicHorizon] = {
    "T+1": "short_term",
    "T+7": "mid_term",
    "T+30": "long_term",
    "24h": "short_term",
    "7d": "mid_term",
    "30d": "long_term",
}


def to_legacy_quant_horizon(value: PublicHorizon) -> LegacyQuantHorizon:
    """Translate a public period to the private inference-service key."""

    return _PUBLIC_TO_LEGACY_QUANT[value]


def public_horizon_payload(value: str) -> PublicHorizon:
    """Normalize a stored/requested legacy value into the public contract."""

    if value in PUBLIC_HORIZONS:
        return cast(PublicHorizon, value)
    try:
        return _LEGACY_TO_PUBLIC[value]
    except KeyError as exc:
        raise ValueError(f"unsupported research horizon: {value}") from exc


def horizon_label(value: PublicHorizon) -> str:
    return HORIZON_LABELS[value]


def horizon_window(value: PublicHorizon) -> str:
    return HORIZON_WINDOWS[value]
