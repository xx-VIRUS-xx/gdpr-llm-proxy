"""
Minimal per-customer usage tracking: a call counter, nothing more.

This deliberately does NOT implement billing, invoicing, rate limiting, or
plan tiers -- it exists only to give a future billing system something to
read from. Unlike the PII mapping store, usage counts are NOT short-TTL:
they are the one piece of state in this system meant to persist (it's a
count, not personal data about the underlying prompts).

Two backends mirror the mapping store's pattern:
  - InMemoryUsageTracker: per-process, lost on restart. Fine for local dev.
  - RedisUsageTracker: durable-ish shared counter via Redis INCR, suitable
    for a real deployment (still not a substitute for a real billing/events
    pipeline -- see README).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from threading import Lock


class UsageTracker(ABC):
    @abstractmethod
    def increment(self, customer_id: str) -> int:
        """Record one proxied call for `customer_id`; return the new total."""

    @abstractmethod
    def get_count(self, customer_id: str) -> int:
        ...


class InMemoryUsageTracker(UsageTracker):
    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def increment(self, customer_id: str) -> int:
        with self._lock:
            self._counts[customer_id] += 1
            return self._counts[customer_id]

    def get_count(self, customer_id: str) -> int:
        return self._counts.get(customer_id, 0)


class RedisUsageTracker(UsageTracker):
    def __init__(self, redis_client=None) -> None:
        if redis_client is None:
            import redis as redis_lib

            from app.config import settings

            redis_client = redis_lib.from_url(settings.redis_url)
        self._redis = redis_client

    def _key(self, customer_id: str) -> str:
        return f"glp:usage:{customer_id}"

    def increment(self, customer_id: str) -> int:
        return int(self._redis.incr(self._key(customer_id)))

    def get_count(self, customer_id: str) -> int:
        value = self._redis.get(self._key(customer_id))
        return int(value) if value is not None else 0


def build_default_tracker() -> UsageTracker:
    from app.config import settings

    if settings.usage_backend == "redis":
        return RedisUsageTracker()
    return InMemoryUsageTracker()
