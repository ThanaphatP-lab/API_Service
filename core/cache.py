from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, wraps
from threading import RLock
from typing import Any


def singleflight_lru_cache(maxsize: int = 1):
    """Cache a loader while allowing only one concurrent cache miss."""

    def decorate(loader: Callable[..., Any]) -> Callable[..., Any]:
        cached = lru_cache(maxsize=maxsize)(loader)
        lock = RLock()

        @wraps(loader)
        def synchronized(*args: Any, **kwargs: Any) -> Any:
            # lru_cache protects its dictionary, but it does not prevent two
            # threads from executing the same uncached GPU loader together.
            with lock:
                return cached(*args, **kwargs)

        synchronized.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
        synchronized.cache_info = cached.cache_info  # type: ignore[attr-defined]
        return synchronized

    return decorate

