import threading
import time
from concurrent.futures import ThreadPoolExecutor

from core.cache import singleflight_lru_cache


def test_singleflight_loader_runs_once_for_concurrent_cache_misses():
    calls = 0
    calls_lock = threading.Lock()

    @singleflight_lru_cache(maxsize=1)
    def loader():
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.02)
        return object()

    with ThreadPoolExecutor(max_workers=6) as executor:
        values = list(executor.map(lambda _: loader(), range(6)))

    assert calls == 1
    assert all(value is values[0] for value in values)
