"""Rule-based news event classifier: category + severity, fully auditable.

Maps a news text onto the event-study taxonomy so the committee's news
analyst can look up historical analogs ("此类事件后 30 天历史均值 …")。Rules,
not ML: the mapping must be explainable in a trace and stable in CI. Texts
that match no category return None -- unclassified news never invents an
analog prior.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Category keyword tables (Chinese + English), checked in order; first
# category with a hit wins ties by score.
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "monetary_policy": [
        "美联储", "联储", "fomc", "fed", "加息", "降息", "利率决议", "点阵图",
        "缩表", "扩表", "qe", "量化宽松", "鲍威尔", "powell", "rate hike",
        "rate cut", "cuts rates", "raises rates", "货币政策", "基点",
    ],
    "inflation": [
        "cpi", "pce", "ppi", "通胀", "通縮", "通缩", "物价", "inflation",
        "核心通胀", "通胀预期", "滞胀",
    ],
    "geopolitics": [
        "战争", "空袭", "袭击", "入侵", "冲突", "制裁", "地缘", "导弹",
        "关税", "贸易战", "war", "strike", "invasion", "sanction", "tariff",
        "军事", "停火", "核设施",
    ],
    "usd": [
        "美元指数", "dxy", "美元走强", "美元走弱", "美元流动性", "汇率",
        "英镑危机", "欧元区危机", "主权评级", "美债收益率",
    ],
    "flows": [
        "央行购金", "黄金储备", "etf", "持仓", "流入", "流出", "避险资金",
        "银行倒闭", "银行危机", "熔断", "流动性危机", "世界黄金协会", "wgc",
    ],
}

# Severity: any high-signal term escalates to "high".
HIGH_SEVERITY_TERMS = [
    "紧急", "熔断", "崩盘", "暴跌", "暴涨", "危机", "倒闭", "违约", "战争",
    "入侵", "空袭", "历史新高", "历史性", "创纪录", "75 个基点", "75bp",
    "100bp", "emergency", "collapse", "crash", "crisis", "war",
]

_CJK_RE = re.compile(r"[一-鿿]")


def _hits(text: str, terms: List[str]) -> List[str]:
    lower = text.lower()
    return [t for t in terms if t in lower]


def classify_news(text: str) -> Optional[Dict[str, Any]]:
    """Return {category, severity, matched} or None when unclassifiable."""
    if not text or not text.strip():
        return None
    lower = text.lower()

    best_category: Optional[str] = None
    best_matches: List[str] = []
    for category, terms in CATEGORY_KEYWORDS.items():
        matches = _hits(lower, terms)
        if len(matches) > len(best_matches):
            best_category = category
            best_matches = matches

    if best_category is None or not best_matches:
        return None

    high_hits = _hits(lower, HIGH_SEVERITY_TERMS)
    severity = "high" if high_hits else "medium"

    return {
        "category": best_category,
        "severity": severity,
        "matched": best_matches[:5] + high_hits[:3],
    }
