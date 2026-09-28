import asyncio
import base64
import io

import pytest
import httpx
from fastapi.testclient import TestClient
from PIL import Image

from core import request_parsing
from core.app_factory import create_app
from core.cache import singleflight_lru_cache
from core.limits import check_batch_limits, check_request_size
from core.readiness import add_readiness_route
from core.service_settings import ServiceURLs
from core.settings import limits_settings, runtime_settings
from shared.contracts import ModelAPIError


def test_rotated_tokens_and_health_exemption(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "current-secret")
    monkeypatch.setenv("INTERNAL_API_TOKEN_PREVIOUS", "previous-secret")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "0")
    app = create_app("auth-test", "fake")

    @app.get("/private")
    def private():
        return {"ok": True}

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    for token in ("current-secret", "previous-secret"):
        assert client.get("/private", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    denied = client.get("/private", headers={"Authorization": "Bearer wrong"})
    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"
    assert denied.headers["WWW-Authenticate"] == "Bearer"


def test_inference_queue_limit_and_slot_release(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "0")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "1")
    monkeypatch.setenv("INFERENCE_QUEUE_TIMEOUT_SECONDS", "0.1")

    async def scenario():
        app = create_app("queue-test", "fake")
        entered, release = asyncio.Event(), asyncio.Event()

        @app.post("/infer")
        async def infer():
            entered.set()
            await release.wait()
            return {"ok": True}

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/infer"))
            try:
                await asyncio.wait_for(entered.wait(), timeout=2)
                busy = await client.post("/infer")
                assert busy.status_code == 503
                assert busy.json()["error"]["code"] == "SERVICE_BUSY"
                assert busy.headers["Retry-After"] == "1"
            finally:
                release.set()
                completed = await first
            assert completed.status_code == 200
            assert (await client.post("/infer")).status_code == 200

    asyncio.run(scenario())


def test_limits_keep_defaults_and_read_environment_on_access(monkeypatch):
    monkeypatch.delenv("MAX_BATCH_IMAGES", raising=False)
    assert limits_settings.batch_images == 64
    check_batch_limits([b"a"] * 64)
    with pytest.raises(ModelAPIError) as caught:
        check_batch_limits([b"a"] * 69)
    assert caught.value.status_code == 413
    assert caught.value.details == [{"received_images": 69, "max_images": 64}]
    monkeypatch.setenv("MAX_BATCH_IMAGES", "70")
    check_batch_limits([b"a"] * 69)
    monkeypatch.setenv("MAX_BATCH_IMAGES", "invalid")
    assert limits_settings.batch_images == 64
    monkeypatch.setenv("MAX_BATCH_IMAGES", "0")
    assert limits_settings.batch_images == 1


def test_batch_total_bytes_and_request_limit_errors(monkeypatch):
    monkeypatch.setenv("MAX_BATCH_IMAGES", "64")
    monkeypatch.setenv("MAX_BATCH_UPLOAD_MB", "1")
    monkeypatch.setenv("MAX_REQUEST_MB", "1")
    with pytest.raises(ModelAPIError) as caught:
        check_batch_limits([b"a" * (1024 * 1024), b"b"])
    assert caught.value.code == "BATCH_TOO_LARGE"
    assert caught.value.details[0]["received_bytes"] == 1024 * 1024 + 1
    check_request_size(str(1024 * 1024))
    for value, status, code in [("invalid", 400, "INVALID_CONTENT_LENGTH"),
                                (str(1024 * 1024 + 1), 413, "PAYLOAD_TOO_LARGE")]:
        with pytest.raises(ModelAPIError) as caught:
            check_request_size(value)
        assert (caught.value.status_code, caught.value.code) == (status, code)


def test_settings_preserve_fallback_and_strict_conversion(monkeypatch):
    monkeypatch.setenv("OCR_RECOGNITION_BATCH_SIZE", "69")
    monkeypatch.setenv("MAX_BATCH_IMAGES", "64")
    assert runtime_settings.recognition_batch_size == 64
    monkeypatch.setenv("GATEWAY_READINESS_TIMEOUT_SECONDS", "invalid")
    assert runtime_settings.gateway_readiness_timeout == 10.0
    monkeypatch.setenv("MAX_UPLOAD_MB", "invalid")
    with pytest.raises(ValueError):
        _ = limits_settings.upload_bytes


def test_urls_are_snapshots_with_existing_defaults(monkeypatch):
    monkeypatch.delenv("REC_SERVICE_URL", raising=False)
    settings = ServiceURLs()
    assert settings.rec_service_url == "http://localhost:8004"
    monkeypatch.setenv("REC_SERVICE_URL", "http://rec:9000")
    assert settings.rec_service_url == "http://localhost:8004"
    assert ServiceURLs().rec_service_url == "http://rec:9000"


def test_failed_second_image_cleans_first_temporary_file(monkeypatch, tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    values = [base64.b64encode(buffer.getvalue()).decode(), base64.b64encode(b"invalid image").decode()]
    original = request_parsing._write_verified_image
    written = []

    def record(data):
        path = original(data)
        written.append(path)
        return path

    monkeypatch.setattr(request_parsing, "_write_verified_image", record)
    monkeypatch.setattr(request_parsing.tempfile, "tempdir", str(tmp_path))

    class FakeRequest:
        headers = {"content-type": "application/json"}

        async def json(self):
            return {"images": values}

    with pytest.raises(ModelAPIError) as caught:
        asyncio.run(request_parsing.parse_image_request(FakeRequest(), multiple=True))
    assert caught.value.code == "INVALID_IMAGE"
    assert len(written) == 1
    assert not written[0].exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("fails", [False, True])
def test_readiness_route_preserves_contract(monkeypatch, fails):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "0")
    app = create_app("core-test", "test-model")
    calls = []

    def loader():
        calls.append(True)
        if fails:
            raise RuntimeError("unavailable")

    add_readiness_route(app, loader)
    response = TestClient(app).get("/api/v1/readiness")
    assert len(calls) == 1
    assert response.status_code == (503 if fails else 200)
    if fails:
        assert response.json()["error"]["code"] == "MODEL_NOT_READY"
    else:
        assert response.json()["data"]["status"] == "ready"
