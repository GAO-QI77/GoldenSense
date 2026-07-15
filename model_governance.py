"""Champion-challenger governance: demote the model when it stops earning trust.

GoldenSense already degrades on *availability* (a tool failed). This adds
degradation on *performance*: if the system's realized directional hit rate
falls below an acceptable band over enough matured calls, governance demotes
to a conservative posture and marks it, instead of quietly staying confident.

The rule-based regime map is always the champion floor -- there is no ML model
that can "take over"; demotion only makes the published stance more cautious
(caps confidence, biases toward 观望) and raises a visible flag. This is a
deterministic, auditable policy, not an online learner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Minimum matured directional calls before governance will act at all.
MIN_SAMPLES = 20
# Hit-rate bands (directional calls only; coin flip = 0.50).
DEMOTE_BELOW = 0.45      # sustained sub-coin-flip -> demote to conservative
WATCH_BELOW = 0.50       # below even chance -> watch, tighten confidence
# Confidence multiplier bounds mirror outcome_tracker (bounded feedback).
MAX_SHIFT = 0.20


@dataclass
class GovernanceVerdict:
    mode: str                       # champion | watch | demoted | insufficient_data
    demoted: bool                   # force conservative posture in the gateway
    confidence_multiplier: float    # bounded scale applied to fused confidence
    force_conservative: bool        # cap stance confidence / bias to 观望
    hit_rate: Optional[float]
    directional_calls: int
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "demoted": self.demoted,
            "confidence_multiplier": round(self.confidence_multiplier, 4),
            "force_conservative": self.force_conservative,
            "hit_rate": self.hit_rate,
            "directional_calls": self.directional_calls,
            "reason": self.reason,
        }


def evaluate_governance(
    calibration_summary: Dict[str, Any],
    *,
    min_samples: int = MIN_SAMPLES,
) -> GovernanceVerdict:
    """Turn a calibration summary (outcome_tracker) into a governance verdict."""
    n = int(calibration_summary.get("directional_calls", 0) or 0)
    hit_rate = calibration_summary.get("hit_rate")

    if hit_rate is None or n < min_samples:
        return GovernanceVerdict(
            mode="insufficient_data",
            demoted=False,
            confidence_multiplier=1.0,
            force_conservative=False,
            hit_rate=hit_rate,
            directional_calls=n,
            reason=f"仅 {n} 次已到期方向判断（需 ≥ {min_samples}），维持规则冠军，不调整。",
        )

    hit_rate = float(hit_rate)
    # Bounded, symmetric confidence feedback around coin-flip.
    edge = max(-1.0, min(1.0, (hit_rate - 0.5) * 2.0))
    multiplier = max(1.0 - MAX_SHIFT, min(1.0 + MAX_SHIFT, 1.0 + edge * MAX_SHIFT))

    if hit_rate < DEMOTE_BELOW:
        return GovernanceVerdict(
            mode="demoted",
            demoted=True,
            confidence_multiplier=multiplier,
            force_conservative=True,
            hit_rate=hit_rate,
            directional_calls=n,
            reason=(
                f"近 {n} 次方向判断命中率 {hit_rate:.0%} 低于 {DEMOTE_BELOW:.0%}，"
                "触发性能降级：收敛为保守观望姿态并标记，等待表现回稳。"
            ),
        )
    if hit_rate < WATCH_BELOW:
        return GovernanceVerdict(
            mode="watch",
            demoted=False,
            confidence_multiplier=multiplier,
            force_conservative=False,
            hit_rate=hit_rate,
            directional_calls=n,
            reason=(
                f"近 {n} 次命中率 {hit_rate:.0%} 低于 50%，进入观察期：下调置信度，暂不降级。"
            ),
        )
    return GovernanceVerdict(
        mode="champion",
        demoted=False,
        confidence_multiplier=multiplier,
        force_conservative=False,
        hit_rate=hit_rate,
        directional_calls=n,
        reason=f"近 {n} 次命中率 {hit_rate:.0%} 达标，维持冠军姿态。",
    )


# Confidence-band de-rating applied when governance says force_conservative.
_BAND_DERATE = {"高": "中", "中": "低", "低": "低"}


def apply_confidence_derate(confidence_band: str, verdict: GovernanceVerdict) -> str:
    """Lower a confidence band one notch when governance forces conservatism."""
    if verdict.force_conservative:
        return _BAND_DERATE.get(confidence_band, confidence_band)
    return confidence_band
