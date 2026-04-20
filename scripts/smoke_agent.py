from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import requests


def _post_json(url: str, payload: Dict[str, Any], timeout_s: float, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout_s, headers=headers)
    resp.raise_for_status()
    return resp.json()


def _get_json(url: str, timeout_s: float, headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    resp = requests.get(url, timeout=timeout_s, headers=headers)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the GoldenSense retail agent flow.")
    parser.add_argument("--agent-base", default="http://localhost:8020")
    parser.add_argument("--market-base", default="http://localhost:8014")
    parser.add_argument("--news-base", default="http://localhost:8016")
    parser.add_argument("--question", default="如果今晚 CPI 高于预期，黄金 24 小时怎么看？")
    parser.add_argument("--risk-profile", default="conservative", choices=["conservative", "balanced", "aggressive"])
    parser.add_argument("--horizon", default="24h", choices=["24h", "7d", "30d"])
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--public-api-key", default=os.environ.get("AGENT_PUBLIC_API_KEY", "dev-public-key"))
    parser.add_argument("--internal-api-key", default=os.environ.get("AGENT_INTERNAL_API_KEY", "dev-internal-key"))
    args = parser.parse_args()

    market_refresh_url = f"{args.market_base}/api/v1/market/snapshot/refresh"
    news_refresh_url = f"{args.news_base}/api/v1/news/refresh"
    analyze_url = f"{args.agent_base}/api/v1/agent/analyze"
    feedback_url = f"{args.agent_base}/api/v1/agent/feedback"
    public_headers = {"X-API-Key": args.public_api_key}
    internal_headers = {"X-API-Key": args.internal_api_key}

    print("1. Refreshing market snapshot...")
    market = _post_json(market_refresh_url, {}, timeout_s=args.timeout)
    print(json.dumps({"latest_price": market["latest_price"], "is_stale": market["is_stale"]}, ensure_ascii=False))

    print("2. Refreshing recent news...")
    news = _post_json(news_refresh_url, {}, timeout_s=args.timeout)
    print(
        json.dumps(
            {
                "news_count": len(news["items"]),
                "freshness_seconds": news["freshness_seconds"],
                "top_sources": [item["source"] for item in news["items"][:3]],
            },
            ensure_ascii=False,
        )
    )

    print("3. Running analyze...")
    analysis = _post_json(
        analyze_url,
        {
            "question": args.question,
            "risk_profile": args.risk_profile,
            "horizon": args.horizon,
            "locale": "zh-CN",
        },
        timeout_s=args.timeout,
        headers=public_headers,
    )
    print(
        json.dumps(
            {
                "analysis_id": analysis["analysis_id"],
                "stance": analysis["summary_card"]["stance"],
                "action": analysis["summary_card"]["action"],
                "risk_banner": analysis["risk_banner"]["title"],
                "horizons": [
                    {
                        "horizon": item["horizon"],
                        "stance": item["stance"],
                        "action": item["action"],
                    }
                    for item in analysis["horizon_forecasts"]
                ],
            },
            ensure_ascii=False,
        )
    )

    print("4. Submitting helpful feedback...")
    feedback = _post_json(
        feedback_url,
        {
            "analysis_id": analysis["analysis_id"],
            "rating": "helpful",
            "comment": "smoke test",
        },
        timeout_s=args.timeout,
        headers=public_headers,
    )
    print(json.dumps(feedback, ensure_ascii=False))

    print("5. Fetching trace...")
    trace = _get_json(
        f"{args.agent_base}/api/v1/agent/traces/{analysis['analysis_id']}",
        timeout_s=args.timeout,
        headers=internal_headers,
    )
    print(
        json.dumps(
            {
                "analysis_id": trace["analysis_id"],
                "tool_calls": len(trace["tool_trace"]),
                "feedback_rating": trace["feedback_rating"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
