import io
import json
import logging

import pytest
from fastapi.testclient import TestClient
from core import audit
from core.app_factory import create_app


@pytest.fixture
def records(monkeypatch):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    monkeypatch.setattr(audit.logger, "handlers", [handler])
    monkeypatch.setattr(audit.logger, "level", logging.INFO)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "synthetic-key")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "0")
    monkeypatch.delenv("MODEL_AUDIT_LOG_FILE", raising=False)
    return lambda: [json.loads(line) for line in stream.getvalue().splitlines()]


def test_auth_and_summary_do_not_log_secrets_or_body(records):
    app = create_app("audit-test", "fake")
    called = []
    @app.post("/work")
    def work():
        called.append(True)
        return {"ok": True}
    client = TestClient(app)
    assert client.post("/work?token=secret-query", content="secret-document").status_code == 401
    assert not called
    assert client.post("/work", headers={"Authorization": "Bearer synthetic-key"}).status_code == 200
    rows = records()
    assert any(r["event"] == "auth_rejected" for r in rows)
    assert any(r["event"] == "auth_accepted" and r["credential_slot"] == "primary" for r in rows)
    summaries = [r for r in rows if r["event"] == "request_completed"]
    assert [r["status"] for r in summaries] == [401, 200]
    assert summaries[0]["request_body_bytes_read"] == 0
    assert summaries[1]["route"] == "/work"
    for value in ("synthetic-key", "secret-query", "secret-document"):
        assert value not in json.dumps(rows)


def test_unobserved_outer_error_is_not_reported_as_zero_byte_success(records):
    app = create_app("audit-test", "fake")
    @app.get("/fail")
    def fail():
        raise RuntimeError("private exception text")
    response = TestClient(app, raise_server_exceptions=False).get(
        "/fail", headers={"X-API-Key": "synthetic-key"})
    assert response.status_code == 500
    summary = [r for r in records() if r["event"] == "request_completed"][-1]
    assert summary["status"] is None
    assert summary["response_complete"] is False
    assert "private exception text" not in json.dumps(records())


def test_upstream_events_redact_url_credentials_and_errors(records):
    @audit.audit_upstream
    def fake(base_url, endpoint, image_paths, *, request_id):
        raise TimeoutError("private-detail")
    with pytest.raises(TimeoutError):
        fake("http://user:password@internal:8000", "/infer?secret=value", [], request_id="req_test")
    rows = records()
    assert [r["event"] for r in rows] == ["upstream_started", "upstream_failed"]
    assert rows[0]["upstream_host"] == "internal"
    assert rows[1]["error_code"] == "TimeoutError"
    assert "password" not in json.dumps(rows)
    assert "private-detail" not in json.dumps(rows)
    assert "secret=value" not in json.dumps(rows)


def test_json_escapes_newlines_and_bounds_strings(records):
    audit.emit("test", path="/x\nforged", value="x" * 2000)
    assert len(records()) == 1
    assert len(records()[0]["value"]) == 1024
