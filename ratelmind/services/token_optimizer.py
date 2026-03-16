import tiktoken
import logging
import json
from enum import Enum
from typing import List, Dict, Any, Optional

class TokenUsageTracker:
    """
    Tracks token usage and cost for LLM interactions.
    """
    
    # Cost per 1k tokens (Input, Output)
    MODEL_COSTS = {
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.00015, 0.0006),
        "claude-3-opus": (0.015, 0.075),
        "claude-3-sonnet": (0.003, 0.015),
        "claude-3-haiku": (0.00025, 0.00125),
    }

    def __init__(self):
        self.usage_log = []
        self.total_cost = 0.0
        self.total_tokens = 0

    def track_usage(self, model: str, prompt_tokens: int, completion_tokens: int, metadata: Dict[str, Any] = None):
        """
        Record token usage for a request.
        """
        cost_in, cost_out = self.MODEL_COSTS.get(model, (0.0, 0.0))
        cost = (prompt_tokens / 1000 * cost_in) + (completion_tokens / 1000 * cost_out)
        
        entry = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": cost,
            "metadata": metadata or {}
        }
        
        self.usage_log.append(entry)
        self.total_cost += cost
        self.total_tokens += entry["total_tokens"]
        
        if cost > 0.1:  # Simple Alert Threshold
            logging.warning(f"High cost request detected: ${cost:.4f} ({model})")
            
        return entry

    def get_summary(self):
        return {
            "total_requests": len(self.usage_log),
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}",
            "avg_cost_per_req": f"${self.total_cost / max(1, len(self.usage_log)):.4f}"
        }

class ContextCompressor:
    """
    Optimizes context by summarization and pruning.
    """
    
    def __init__(self, encoding_name="cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def truncate_context(self, text: str, max_tokens: int = 2000) -> str:
        """
        Hard truncation to fit token limit.
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens]) + "..."

    def summarize_news(self, news_items: List[Dict], max_items: int = 5) -> str:
        """
        Compress news list into a concise summary prompt.
        Prioritizes by score if available.
        """
        # Sort by importance/score if available
        sorted_news = sorted(news_items, key=lambda x: x.get('score', 0) or 0, reverse=True)
        
        summary = []
        for item in sorted_news[:max_items]:
            title = item.get('title', 'No Title')
            # Extract only first sentence of summary to save tokens
            desc = item.get('summary', '').split('.')[0]
            score = item.get('score', 0)
            summary.append(f"- [{score:.1f}] {title}: {desc}")
            
        return "\n".join(summary)

    def format_prompt_efficiently(self, system_prompt: str, user_input: str, context: str) -> List[Dict]:
        """
        Constructs a token-optimized message list.
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nTask: {user_input}"}
        ]

class ModelRouter:
    """
    Routes requests to appropriate models based on complexity.
    """
    
    class TaskComplexity(Enum):
        SIMPLE = "simple"   # Fact extraction, formatting
        MEDIUM = "medium"   # Summarization, sentiment analysis
        COMPLEX = "complex" # Reasoning, creative writing, coding

    def select_model(self, complexity: TaskComplexity) -> str:
        if complexity == self.TaskComplexity.SIMPLE:
            return "gpt-4o-mini"
        elif complexity == self.TaskComplexity.MEDIUM:
            return "gpt-4o-mini" # Aggressive optimization: try mini first
        else:
            return "gpt-4o"

# Example Usage
if __name__ == "__main__":
    tracker = TokenUsageTracker()
    compressor = ContextCompressor()
    router = ModelRouter()

    # Simulate News Data
    news = [
        {"title": "Fed Hikes Rates", "summary": "The Federal Reserve increased rates by 25bps. Market reacts negatively.", "score": -0.8},
        {"title": "Gold Prices Soar", "summary": "Gold hits all-time high amidst uncertainty. Investors flock to safety.", "score": 0.9},
        {"title": "Tech Stocks Rally", "summary": "Nasdaq up 2%. AI sector leads gains.", "score": 0.5},
        # ... 50 more items ...
    ]
    
    # 1. Compress Context
    compressed_ctx = compressor.summarize_news(news, max_items=3)
    print(f"Compressed Context ({compressor.count_tokens(compressed_ctx)} tokens):\n{compressed_ctx}\n")

    # 2. Route Model
    model = router.select_model(ModelRouter.TaskComplexity.MEDIUM)
    print(f"Selected Model: {model}")

    # 3. Simulate Request & Track
    # (Mocking actual API call response usage)
    tracker.track_usage(model, prompt_tokens=150, completion_tokens=50)
    
    print("\nToken Usage Summary:")
    print(json.dumps(tracker.get_summary(), indent=2))
