from news_provenance import classify_news_source
from service_contracts import NewsEventItem


def _item(*, source: str, url: str | None) -> NewsEventItem:
    return NewsEventItem(
        event_id="news-1",
        published_at="2026-08-15T00:00:00Z",
        title="Gold market update",
        summary="A source-specific market update.",
        source=source,
        normalized_event="gold",
        sentiment_score=0,
        importance=0.8,
        categories=["macro"],
        url=url,
    )


def test_official_federal_reserve_release_is_primary():
    classified = classify_news_source(
        _item(
            source="Federal Reserve",
            url="https://www.federalreserve.gov/newsevents/pressreleases/test.htm",
        )
    )

    assert classified.source_tier == "primary"
    assert classified.source_authority == "Federal Reserve"
    assert classified.is_primary_source is True


def test_official_subdomain_is_primary_but_lookalike_domain_is_not():
    official = classify_news_source(
        _item(source="CFTC", url="https://www.cftc.gov/MarketReports/test.htm")
    )
    lookalike = classify_news_source(
        _item(source="CFTC report mirror", url="https://cftc.gov.example.com/story")
    )

    assert official.source_tier == "primary"
    assert lookalike.source_tier == "secondary"
    assert lookalike.is_primary_source is False


def test_wire_story_is_secondary_not_primary():
    classified = classify_news_source(
        _item(source="wire", url="https://example.com/story")
    )

    assert classified.source_tier == "secondary"
    assert classified.source_authority == "wire"


def test_market_authorities_and_central_banks_are_primary_by_hostname():
    sources = [
        ("https://www.cmegroup.com/markets/metals/precious/gold.html", "CME Group"),
        ("https://www.gold.org/goldhub/research", "World Gold Council"),
        ("https://www.ecb.europa.eu/press/html/index.en.html", "European Central Bank"),
        ("https://www.bankofengland.co.uk/news", "Bank of England"),
        ("https://www.pbc.gov.cn/en/3688006/index.html", "People's Bank of China"),
    ]

    for url, authority in sources:
        classified = classify_news_source(_item(source="untrusted display", url=url))
        assert classified.source_tier == "primary", url
        assert classified.source_authority == authority
        assert classified.is_primary_source is True


def test_fxstreet_remains_secondary_even_when_title_mentions_an_authority():
    classified = classify_news_source(
        _item(source="Federal Reserve release via FXStreet", url="https://www.fxstreet.com/news/story")
    )

    assert classified.source_tier == "secondary"
    assert classified.is_primary_source is False


def test_synthetic_fallback_is_never_presented_as_news_source():
    classified = classify_news_source(
        _item(source="synthetic_fallback", url=None)
    )

    assert classified.source_tier == "synthetic"
    assert classified.is_primary_source is False
