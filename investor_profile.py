"""Investor profile & personalized narrative models.

The profile travels with each request and is never persisted server-side:
personalization without a user database is both the lighter build and the
stronger privacy stance. Strict validation (``extra="forbid"``) keeps the
surface honest -- unknown fields are rejected, not silently swallowed.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

RiskTolerance = Literal["conservative", "balanced", "aggressive"]
Horizon = Literal["short", "mid", "long"]
Experience = Literal["novice", "experienced", "professional"]
LiquidityNeed = Literal["low", "medium", "high"]
LeverageAttitude = Literal["none", "low", "medium", "high"]
InvestmentGoal = Literal[
    "capital_preservation", "income", "event_trade", "trend_following", "speculation"
]


class InvestorProfile(BaseModel):
    """Unified investor profile: a 4-field core plus an optional advanced
    layer. The core alone is a complete request (backward compatible with the
    original schema); advanced fields, when present, unlock additional
    deterministic risk rules -- they never change the reference range itself.
    """

    model_config = ConfigDict(extra="forbid")

    # Core layer (required)
    risk_tolerance: RiskTolerance
    horizon: Horizon
    current_gold_pct: float = Field(ge=0.0, le=100.0)
    experience: Experience

    # Advanced layer (optional)
    max_drawdown_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    liquidity_need: Optional[LiquidityNeed] = None
    leverage_attitude: Optional[LeverageAttitude] = None
    investment_goal: Optional[InvestmentGoal] = None


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
