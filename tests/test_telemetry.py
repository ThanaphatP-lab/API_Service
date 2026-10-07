import asyncio
import logging
import io

from core.telemetry import RequestTelemetry, current_request, timed_stage


def test_access_log_copied_to_debug_once(tmp_path, monkeypatch):
    from core.telemetry import configure_access_logging
    names = ("uvicorn.error", "uvicorn.access", "model_api")
    logs = {name: logging.Logger(name, logging.INFO) for name in names}
    console = io.StringIO()
    logs["uvicorn.access"].addHandler(logging.StreamHandler(console))
    original_get_logger = logging.getLogger
    monkeypatch.setattr(logging, "getLogger", lambda name=None: logs[name] if name in logs else original_get_logger(name))
    monkeypatch.setenv("MODEL_DEBUG_LOG_FILE", str(tmp_path / "debug.log"))
    monkeypatch.setenv("API_ACCESS_LOG", "true")
    try:
        configure_access_logging()
        configure_access_logging()
        logs["uvicorn.access"].info('%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:60404", "POST",
            "/api/v1/text-recognition-batches?version=6&model=thai_ft_v1", "1.1", 200)
        content = (tmp_path / "debug.log").read_text(encoding="utf-8")
        assert content.count("127.0.0.1:60404") == 1
        assert "version=6&model=thai_ft_v1" in content
        assert 'HTTP/1.1" 200 OK' in content
        assert "\x1b" not in content
        assert console.getvalue().count("127.0.0.1:60404") == 1
        handlers = logs["uvicorn.access"].handlers
        assert handlers[-1] is logs["uvicorn.error"].handlers[-1]
    finally:
        for handler in logs["uvicorn.error"].handlers:
            handler.close()


def test_debug_file_is_rotated_and_console_stays_info(tmp_path, monkeypatch):
    from core.telemetry import configure_debug_file
    log = logging.Logger("isolated")
    console = io.StringIO()
    log.addHandler(logging.StreamHandler(console))
    path = tmp_path / "service.debug.log"
    monkeypatch.setenv("MODEL_DEBUG_LOG_FILE", str(path))
    try:
        configure_debug_file(log, logging.INFO)
        configure_debug_file(log, logging.INFO)
        assert len(log.handlers) == 2
        file_handler = log.handlers[-1]
        assert file_handler.maxBytes == 20 * 1024 * 1024
        assert file_handler.backupCount == 3
        log.debug("detail")
        log.info("summary")
        assert console.getvalue() == "summary\n"
        assert "detail" in path.read_text(encoding="utf-8")
        assert "summary" in path.read_text(encoding="utf-8")
        file_handler.maxBytes = 100
        for _ in range(20):
            log.debug("rotation test record")
        assert len(list(tmp_path.glob("service.debug.log*"))) == 4
    finally:
        for handler in log.handlers:
            handler.close()


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
    assert "in=" in caplog.text and "out=" in caplog.text
    assert "request_received" not in caplog.text
    assert "resources=" not in caplog.text
    assert len(caplog.records) == 1
    assert "hidden" not in caplog.text
    assert current_request.get() == "-"


def test_stage_logs_failure(caplog):
    @timed_stage("test")
    def fail():
        raise ValueError("no")
    import pytest
    with caplog.at_level(logging.DEBUG, logger="uvicorn.error"), pytest.raises(ValueError):
        fail()
    assert "outcome=error" in caplog.text


def test_stage_accumulates_without_info_noise(caplog):
    from core.telemetry import request_timings
    timings = {}
    token = request_timings.set(timings)
    try:
        @timed_stage("inference_including_load_and_adaptation")
        def work(): return 1
        with caplog.at_level(logging.INFO, logger="uvicorn.error"):
            assert work() == 1
        assert timings["inference_including_load_and_adaptation"] >= 0
        assert not caplog.records
    finally:
        request_timings.reset(token)


def test_access_log_filter_is_scoped_and_opt_out(monkeypatch):
    from core.telemetry import ManagedAccessFilter
    f = ManagedAccessFilter()
    monkeypatch.delenv("API_ACCESS_LOG", raising=False)
    assert f.filter(None)
    token = current_request.set("req_test")
    try:
        assert f.filter(None)
        monkeypatch.setenv("API_ACCESS_LOG", "false")
        assert not f.filter(None)
        monkeypatch.setenv("API_ACCESS_LOG", "true")
        assert f.filter(None)
    finally:
        current_request.reset(token)
