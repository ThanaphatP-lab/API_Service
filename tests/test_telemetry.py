import asyncio
import logging

from core.telemetry import RequestTelemetry, current_request, timed_stage


def test_request_logs_rejection_and_body_without_sensitive_query(caplog):
    async def run():
        async def app(scope, receive, send):
            assert current_request.get() == "req_test"
            await receive()
            await send({"type": "http.response.start", "status": 413, "headers": []})
            await send({"type": "http.response.body", "body": b"bad"})
        async def receive():
            return {"type": "http.request", "body": b"abc"}
        async def send(message):
            pass
        await RequestTelemetry(app, "test")({"type": "http", "method": "POST", "path": "/test",
            "query_string": b"secret=hidden", "headers": [(b"x-request-id", b"req_test")]}, receive, send)
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        asyncio.run(run())
    assert "status=413" in caplog.text
    assert "received_bytes=3 sent_bytes=3" in caplog.text
    assert "hidden" not in caplog.text
    assert current_request.get() == "-"


def test_stage_logs_failure(caplog):
    @timed_stage("test")
    def fail():
        raise ValueError("no")
    import pytest
    with caplog.at_level(logging.INFO, logger="uvicorn.error"), pytest.raises(ValueError):
        fail()
    assert "outcome=error" in caplog.text
