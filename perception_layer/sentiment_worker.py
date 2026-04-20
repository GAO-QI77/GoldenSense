from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from perception_layer.state_manager import NEWS_QUEUE_KEY, StateManager


@dataclass(frozen=True)
class NewsItem:
    published_at: str
    title: str
    body: str


class FinBertScorer:
    def __init__(self, model_id: str = "ProsusAI/finbert"):
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._model.eval()

    @torch.no_grad()
    def score(self, text: str) -> float:
        inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        logits = self._model(**inputs).logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(float)
        label_to_index = {str(label).lower(): int(idx) for idx, label in self._model.config.id2label.items()}
        pos_idx = label_to_index.get("positive")
        neg_idx = label_to_index.get("negative")
        if pos_idx is None or neg_idx is None:
            return 0.0
        p_pos = float(probs[pos_idx])
        p_neg = float(probs[neg_idx])
        score = p_pos - p_neg
        return float(np.clip(score, -1.0, 1.0))


def classify_dimension(title: str, body: str) -> str:
    text = f"{title} {body}".lower()
    monetary_keywords = [
        "fomc",
        "fed",
        "powell",
        "interest rate",
        "rate hike",
        "rate cut",
        "tightening",
        "easing",
        "central bank",
        "quantitative",
    ]
    macro_keywords = [
        "cpi",
        "ppi",
        "inflation",
        "gdp",
        "unemployment",
        "payroll",
        "nonfarm",
        "nfp",
        "retail sales",
        "pmis",
        "pmi",
    ]
    geo_keywords = [
        "war",
        "conflict",
        "missile",
        "sanction",
        "invasion",
        "geopolitical",
        "attack",
        "ceasefire",
        "terror",
    ]

    if any(k in text for k in monetary_keywords):
        return "Monetary_Policy"
    if any(k in text for k in geo_keywords):
        return "Geopolitics"
    if any(k in text for k in macro_keywords):
        return "Macro_Economy"
    return "Macro_Economy"


def parse_news_item(payload: str) -> Optional[NewsItem]:
    try:
        obj = json.loads(payload)
        published_at = str(obj.get("published_at", ""))
        title = str(obj.get("title", ""))
        body = str(obj.get("body", ""))
        if not title:
            return None
        return NewsItem(published_at=published_at, title=title, body=body)
    except Exception:
        return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--model-id", default="ProsusAI/finbert")
    parser.add_argument("--queue-key", default=NEWS_QUEUE_KEY)
    parser.add_argument("--block-seconds", type=int, default=5)
    args = parser.parse_args()

    sm = StateManager(redis_url=args.redis_url)
    if not sm.ping():
        raise SystemExit("redis ping failed")

    scorer = FinBertScorer(model_id=args.model_id)

    while True:
        payload = sm.pop_news(queue_key=args.queue_key, timeout_s=args.block_seconds)
        if payload is None:
            continue
        news = parse_news_item(payload)
        if news is None:
            continue
        dimension = classify_dimension(news.title, news.body)
        text = f"{news.title}\n{news.body}".strip()
        score = scorer.score(text)
        headline = news.title.strip()
        sm.update_sentiment(dimension=dimension, score=score, headline=headline)
        now = datetime.now(timezone.utc).isoformat()
        print(f"{now} dim={dimension} score={score:.3f} title={headline}")


if __name__ == "__main__":
    main()
