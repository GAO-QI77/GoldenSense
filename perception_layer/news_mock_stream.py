from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Iterator, Optional

from perception_layer.state_manager import NEWS_QUEUE_KEY, StateManager


def iter_jsonl(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield s


def normalize_item(raw_line: str) -> Optional[str]:
    try:
        obj = json.loads(raw_line)
        published_at = str(obj.get("published_at", ""))
        title = str(obj.get("title", "")).strip()
        body = str(obj.get("body", "")).strip()
        if not title:
            return None
        return json.dumps({"published_at": published_at, "title": title, "body": body}, ensure_ascii=False)
    except Exception:
        return None


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--queue-key", default=NEWS_QUEUE_KEY)
    parser.add_argument("--data-path", default=str(Path(__file__).with_name("news_mock_data.jsonl")))
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    sm = StateManager(redis_url=args.redis_url)
    if not sm.ping():
        raise SystemExit("redis ping failed")

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise SystemExit(f"data file not found: {data_path}")

    while True:
        pushed = 0
        for raw_line in iter_jsonl(data_path):
            payload = normalize_item(raw_line)
            if payload is None:
                continue
            sm.push_news(payload, queue_key=args.queue_key)
            pushed += 1
            try:
                obj = json.loads(payload)
                print(f"push #{pushed} {obj.get('published_at','')} {obj.get('title','')}")
            except Exception:
                print(f"push #{pushed}")
            await asyncio.sleep(args.interval_seconds)

        if not args.loop:
            break


if __name__ == "__main__":
    asyncio.run(main())

