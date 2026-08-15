"""Deterministic provenance labels for market-news presentation.

Primary status is granted only from the parsed hostname. Titles, summaries and
publisher display names are untrusted text and never upgrade a source tier.
"""

from __future__ import annotations

from urllib.parse import urlparse

from service_contracts import NewsEventItem


_PRIMARY_AUTHORITIES: tuple[tuple[str, str], ...] = (
    ("federalreserve.gov", "Federal Reserve"),
    ("bls.gov", "U.S. Bureau of Labor Statistics"),
    ("bea.gov", "U.S. Bureau of Economic Analysis"),
    ("treasury.gov", "U.S. Department of the Treasury"),
    ("cftc.gov", "U.S. Commodity Futures Trading Commission"),
    ("cmegroup.com", "CME Group"),
    ("gold.org", "World Gold Council"),
    ("stlouisfed.org", "Federal Reserve Bank of St. Louis"),
    ("bis.org", "Bank for International Settlements"),
    ("ecb.europa.eu", "European Central Bank"),
    ("bankofengland.co.uk", "Bank of England"),
    ("pbc.gov.cn", "People's Bank of China"),
    ("boj.or.jp", "Bank of Japan"),
)


def _matches_domain(hostname: str, registered_domain: str) -> bool:
    return hostname == registered_domain or hostname.endswith(f".{registered_domain}")


def classify_news_source(item: NewsEventItem) -> NewsEventItem:
    """Return a copy with a presentation-safe source tier and authority."""

    if item.source == "synthetic_fallback":
        return item.model_copy(
            update={
                "source_tier": "synthetic",
                "source_authority": None,
                "is_primary_source": False,
            }
        )

    hostname = ""
    if item.url:
        try:
            parsed = urlparse(item.url)
            if parsed.scheme in {"https", "http"}:
                hostname = (parsed.hostname or "").lower().rstrip(".")
        except ValueError:
            hostname = ""

    for domain, authority in _PRIMARY_AUTHORITIES:
        if _matches_domain(hostname, domain):
            return item.model_copy(
                update={
                    "source_tier": "primary",
                    "source_authority": authority,
                    "is_primary_source": True,
                }
            )

    return item.model_copy(
        update={
            "source_tier": "secondary" if item.source else "unknown",
            "source_authority": item.source or None,
            "is_primary_source": False,
        }
    )


def classify_news_items(items: list[NewsEventItem]) -> list[NewsEventItem]:
    return [classify_news_source(item) for item in items]
