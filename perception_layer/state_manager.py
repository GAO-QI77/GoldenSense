from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import redis


LIVE_STATE_KEY = "golden_sense:live_state"
NEWS_QUEUE_KEY = "golden_sense:news_queue"


@dataclass(frozen=True)
class LiveState:
    latest_price: Optional[float]
    price_timestamp: Optional[str]
    macro_sentiment_score: Optional[float]
    fed_sentiment_score: Optional[float]
    last_news_headline: Optional[str]


class StateManager:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        macro_ema_alpha: float = 0.2,
        fed_ema_alpha: float = 0.2,
    ):
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._macro_ema_alpha = _clamp01(macro_ema_alpha)
        self._fed_ema_alpha = _clamp01(fed_ema_alpha)

    def ping(self) -> bool:
        return bool(self._redis.ping())

    def get_live_state(self) -> LiveState:
        raw = self._redis.hgetall(LIVE_STATE_KEY)
        return LiveState(
            latest_price=_to_float(raw.get("latest_price")),
            price_timestamp=raw.get("price_timestamp"),
            macro_sentiment_score=_to_float(raw.get("macro_sentiment_score")),
            fed_sentiment_score=_to_float(raw.get("fed_sentiment_score")),
            last_news_headline=raw.get("last_news_headline"),
        )

    def set_fields(self, fields: Dict[str, str]) -> None:
        if not fields:
            return
        self._redis.hset(LIVE_STATE_KEY, mapping=fields)

    def update_sentiment(self, *, dimension: str, score: float, headline: str) -> None:
        score_clamped = _clamp(score, low=-1.0, high=1.0)
        if dimension == "Monetary_Policy":
            prev = _to_float(self._redis.hget(LIVE_STATE_KEY, "fed_sentiment_score"))
            alpha = self._fed_ema_alpha
            smoothed = _ema(prev=prev, new=score_clamped, alpha=alpha)
            updates = {"fed_sentiment_score": f"{smoothed:.6f}"}
        else:
            prev = _to_float(self._redis.hget(LIVE_STATE_KEY, "macro_sentiment_score"))
            alpha = self._macro_ema_alpha
            smoothed = _ema(prev=prev, new=score_clamped, alpha=alpha)
            updates = {"macro_sentiment_score": f"{smoothed:.6f}"}
        updates["last_news_headline"] = headline
        updates["price_timestamp"] = datetime.now(timezone.utc).isoformat()
        self.set_fields(updates)

    def push_news(self, payload: str, *, queue_key: str = NEWS_QUEUE_KEY) -> None:
        self._redis.lpush(queue_key, payload)

    def pop_news(self, *, queue_key: str = NEWS_QUEUE_KEY, timeout_s: int = 5) -> Optional[str]:
        item = self._redis.brpop(queue_key, timeout=timeout_s)
        if item is None:
            return None
        _, payload = item
        return payload


def _to_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None


def _ema(*, prev: Optional[float], new: float, alpha: float) -> float:
    if prev is None:
        return new
    return (1.0 - alpha) * prev + alpha * new


def _clamp01(x: float) -> float:
    return _clamp(x, low=0.0, high=1.0)


def _clamp(x: float, *, low: float, high: float) -> float:
    return max(low, min(high, x))
    try:
        return float(val)
    except ValueError:
        return None


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    sm = StateManager(redis_url=args.redis_url)
    if not sm.ping():
        raise SystemExit("redis ping failed")

    if args.watch:
        while True:
            s = sm.get_live_state()
            print(s)
            time.sleep(args.interval)
    else:
        print(sm.get_live_state())


if __name__ == "__main__":
    main()
