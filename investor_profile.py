"""Investor profile & personalized narrative models.

The profile travels with each request and is never persisted server-side:
personalization without a user database is both the lighter build and the
stronger privacy stance. Strict validation (``extra="forbid"``) keeps the
surface honest -- unknown fields are rejected, not silently swallowed.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

RiskTolerance = Literal["conservative", "balanced", "aggressive"]
Horizon = Literal["short", "mid", "long"]
Experience = Literal["novice", "experienced", "professional"]


class InvestorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_tolerance: RiskTolerance
    horizon: Horizon
    current_gold_pct: float = Field(ge=0.0, le=100.0)
    experience: Experience


class PersonalNarrative(BaseModel):
    """The LLM-polished (or deterministic-draft) personalized narrative.

    Text only: every number these fields mention must already exist in the
    facts payload -- the narrative critic enforces that after generation.
    """

    model_config = ConfigDict(extra="forbid")

    overview: str
    position_analysis: str
    risk_notes: List[str]
    horizon_note: str
    disclaimer: str
