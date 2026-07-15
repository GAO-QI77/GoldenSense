"""Lightweight in-process metrics for the agent gateway.

No external APM dependency -- a thread-safe registry of per-route request
counts, latency (count/sum/max/p95-ish via reservoir), status classes, plus
domain counters (degradation flags, governance demotions). Exposed read-only
at /metrics so ops can watch the new endpoints without wiring Prometheus.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Dict, List


class MetricsRegistry:
    def __init__(self, *, reservoir_size: int = 200):
        self._lock = threading.Lock()
        self._reservoir_size = reservoir_size
        self._counts: Dict[str, int] = defaultdict(int)
        self._errors: Dict[str, int] = defaultdict(int)
        self._latency_sum: Dict[str, float] = defaultdict(float)
        self._latency_max: Dict[str, float] = defaultdict(float)
        self._latency_samples: Dict[str, List[float]] = defaultdict(list)
        self._status_class: Dict[str, int] = defaultdict(int)
        self._domain: Dict[str, int] = defaultdict(int)
        self._started = time.time()

    def record_request(self, route: str, *, status_code: int, elapsed_ms: float) -> None:
        with self._lock:
            self._counts[route] += 1
            if status_code >= 500:
                self._errors[route] += 1
            self._status_class[f"{status_code // 100}xx"] += 1
            self._latency_sum[route] += elapsed_ms
            if elapsed_ms > self._latency_max[route]:
                self._latency_max[route] = elapsed_ms
            samples = self._latency_samples[route]
            samples.append(elapsed_ms)
            if len(samples) > self._reservoir_size:
                del samples[0]

    def incr(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._domain[name] += value

    @staticmethod
    def _percentile(samples: List[float], q: float) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        idx = min(len(ordered) - 1, int(q * len(ordered)))
        return round(ordered[idx], 2)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            routes: Dict[str, Any] = {}
            for route, count in self._counts.items():
                samples = self._latency_samples[route]
                routes[route] = {
                    "count": count,
                    "errors": self._errors[route],
                    "error_rate": round(self._errors[route] / count, 4) if count else 0.0,
                    "latency_ms_avg": round(self._latency_sum[route] / count, 2) if count else 0.0,
                    "latency_ms_max": round(self._latency_max[route], 2),
                    "latency_ms_p95": self._percentile(samples, 0.95),
                }
            return {
                "uptime_seconds": int(time.time() - self._started),
                "routes": routes,
                "status_classes": dict(self._status_class),
                "domain_counters": dict(self._domain),
            }
