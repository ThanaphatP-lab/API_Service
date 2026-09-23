from shared.rate_limit import InMemoryRateLimiter


def test_sliding_window_allows_then_rejects(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "10")
    limiter = InMemoryRateLimiter()

    assert limiter.check("client", now=0).allowed is True
    assert limiter.check("client", now=1).allowed is True
    rejected = limiter.check("client", now=2)

    assert rejected.allowed is False
    assert rejected.remaining == 0
    assert rejected.reset_after == 8
    assert limiter.check("client", now=11).allowed is True


def test_zero_limit_disables_limiter(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "0")
    limiter = InMemoryRateLimiter()

    assert limiter.check("client").allowed is True
    assert limiter.enabled is False
