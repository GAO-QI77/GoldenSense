"""Public-language policy for research outputs.

GoldenSense is an investment-research product, not an order-routing or
personal portfolio management service.  This module provides one deterministic
policy used by every narrator before text reaches a public response.
"""

from __future__ import annotations

import re
from typing import Iterable


_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "target_position",
        re.compile(r"(?:目标|建议)(?:黄金)?(?:仓位|持仓|暴露)(?:约|为|到)?\s*\d+(?:\.\d+)?\s*%?"),
    ),
    (
        "trade_directive",
        re.compile(r"买入|卖出|加仓|减仓|满仓|清仓|开仓|平仓|追涨|杀跌"),
    ),
    ("position_advice", re.compile(r"持仓建议|仓位建议|下单建议")),
)


def find_directive_language(texts: Iterable[str]) -> list[dict[str, str]]:
    """Return one auditable violation per matched policy class."""

    joined = "\n".join(str(text) for text in texts if text)
    violations: list[dict[str, str]] = []
    for code, pattern in _RULES:
        match = pattern.search(joined)
        if match:
            violations.append({"code": code, "match": match.group(0)})
    return violations


def sanitize_research_language(text: str) -> str:
    """Rewrite common execution directives into scenario-research language."""

    value = str(text or "")
    value = _RULES[0][1].sub("黄金风险参考区间需结合个人承受力评估", value)
    value = re.sub(r"持仓建议|仓位建议|下单建议", "风险观察清单", value)
    value = re.sub(r"买入|加仓|开仓|追涨", "关注上行情景", value)
    value = re.sub(r"卖出|减仓|清仓|平仓|杀跌", "关注下行情景", value)
    value = value.replace("满仓", "高风险集中暴露")
    return value
