from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass

from fastapi import Request
from core.settings import RateLimitSettings, runtime_settings


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_after: int


class InMemoryRateLimiter:
    """A process-local sliding-window limiter for a single API worker."""

    def __init__(self) -> None:
        settings = RateLimitSettings()
        self.limit = settings.requests
        self.window = settings.window_seconds
        self.max_keys = settings.max_keys
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def check(self, key: str, now: float | None = None) -> RateLimitDecision:
        if not self.enabled:
            return RateLimitDecision(True, 0, 0, 0)
        current = time.monotonic() if now is None else now
        cutoff = current - self.window
        with self._lock:
            events = self._requests.get(key)
            if events is None:
                if len(self._requests) >= self.max_keys:
                    self._requests.popitem(last=False)
                events = deque()
                self._requests[key] = events
            else:
                self._requests.move_to_end(key)
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                reset_after = max(1, int(events[0] + self.window - current + 0.999))
                return RateLimitDecision(False, self.limit, 0, reset_after)
            events.append(current)
            remaining = max(0, self.limit - len(events))
            reset_after = max(1, int(events[0] + self.window - current + 0.999))
            return RateLimitDecision(True, self.limit, remaining, reset_after)


def client_key(request: Request) -> str:
    if runtime_settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def rate_limit_headers(decision: RateLimitDecision) -> dict[str, str]:
    if decision.limit <= 0:
        return {}
    return {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
        "X-RateLimit-Reset": str(decision.reset_after),
    }
