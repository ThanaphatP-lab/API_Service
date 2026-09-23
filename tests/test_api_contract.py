import base64
import io

import pytest

from fastapi import Request
from fastapi.testclient import TestClient
from PIL import Image

from shared.api import create_app, parse_image_request
from shared.contracts import ModelAPIError, success_response


def _png_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _test_app():
    app = create_app("Contract test", "fake", service_name="contract-test")

    @app.post("/api/v1/images")
    async def images(request: Request):
        image = await parse_image_request(request)
        try:
            return success_response(request, {"size": image.path.stat().st_size}, service="contract-test", model="fake")
        finally:
            image.cleanup()

    return app


def test_json_base64_success_contract(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10")
    client = TestClient(_test_app())

    response = client.post(
        "/api/v1/images",
        json={"image": _png_data_url()},
        headers={"X-Request-ID": "req_test123"},
    )

    assert response.status_code == 200
    assert response.json()["meta"]["request_id"] == "req_test123"
    assert response.json()["meta"]["api_version"] == "v1"
    assert response.headers["X-Request-ID"] == "req_test123"


def test_multipart_image_success(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10")
    client = TestClient(_test_app())
    raw = base64.b64decode(_png_data_url().split(",", 1)[1])

    response = client.post(
        "/api/v1/images",
        files={"image": ("sample.png", raw, "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["data"]["size"] == len(raw)


def test_invalid_base64_has_debuggable_error(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10")
    client = TestClient(_test_app())

    response = client.post("/api/v1/images", json={"image": "not-base64!"})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "INVALID_IMAGE_BASE64"
    assert error["request_id"].startswith("req_")
    assert error["details"][0]["field"] == "image"


def test_rate_limit_returns_429_and_headers(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    client = TestClient(_test_app())

    assert client.post("/api/v1/images", json={"image": _png_data_url()}).status_code == 200
    response = client.post("/api/v1/images", json={"image": _png_data_url()})

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert response.headers["Retry-After"]
    assert response.headers["X-RateLimit-Remaining"] == "0"


def test_protected_service_accepts_bearer_and_api_key_headers(monkeypatch):
    token = "test-token-with-more-than-thirty-two-characters"
    previous_token = "previous-token-with-more-than-thirty-two-characters"
    monkeypatch.setenv("INTERNAL_API_TOKEN", token)
    monkeypatch.setenv("INTERNAL_API_TOKEN_PREVIOUS", previous_token)
    client = TestClient(_test_app())

    unauthenticated = client.post("/api/v1/images", json={"image": _png_data_url()})
    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"

    bearer = client.post(
        "/api/v1/images",
        json={"image": _png_data_url()},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert bearer.status_code == 200

    api_key = client.post(
        "/api/v1/images",
        json={"image": _png_data_url()},
        headers={"X-API-Key": token},
    )
    assert api_key.status_code == 200

    rotating = client.post(
        "/api/v1/images",
        json={"image": _png_data_url()},
        headers={"Authorization": f"Bearer {previous_token}"},
    )
    assert rotating.status_code == 200


def test_development_docs_open_and_describe_auth(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-token-with-more-than-thirty-two-characters")
    client = TestClient(_test_app())

    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
    assert schema["components"]["securitySchemes"]["ApiKeyAuth"]["name"] == "X-API-Key"


def test_request_size_is_rejected_before_image_processing(monkeypatch):
    monkeypatch.setenv("MAX_REQUEST_MB", "1")
    client = TestClient(_test_app())

    response = client.post(
        "/api/v1/images",
        content=b"{}",
        headers={"Content-Type": "application/json", "Content-Length": str(2 * 1024 * 1024)},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_image_pixel_limit_is_enforced(monkeypatch):
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "8")
    client = TestClient(_test_app())

    response = client.post("/api/v1/images", json={"image": _png_data_url()})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_DIMENSIONS_TOO_LARGE"


def test_public_api_redacts_internal_error_details(monkeypatch):
    monkeypatch.delenv("DEBUG_ERRORS", raising=False)
    app = create_app("Public test", "fake", service_name="public-test", public_api=True)

    @app.get("/api/v1/failure")
    def failure():
        raise ModelAPIError(
            503,
            "UPSTREAM_UNAVAILABLE",
            "The upstream is unavailable.",
            details=[{"upstream": "http://127.0.0.1:8003", "reason": "private path", "upstream_code": "DOWN"}],
        )

    response = TestClient(app).get("/api/v1/failure")

    assert response.status_code == 503
    detail = response.json()["error"]["details"][0]
    assert "upstream" not in detail
    assert "reason" not in detail
    assert detail["upstream_code"] == "DOWN"


def test_success_response_has_security_headers(monkeypatch):
    client = TestClient(_test_app())

    response = client.post("/api/v1/images", json={"image": _png_data_url()})

    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_production_rejects_placeholder_secret(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "replace-with-at-least-32-random-characters")

    with pytest.raises(RuntimeError, match="non-placeholder secret"):
        _test_app()


def test_production_requires_outbound_service_secret(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("MODEL_GATEWAY_API_KEY", "public-secret-with-more-than-thirty-two-characters")
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="INTERNAL_API_TOKEN"):
        create_app(
            "Gateway test",
            "fake",
            auth_token_env="MODEL_GATEWAY_API_KEY",
            required_auth_token_envs=("INTERNAL_API_TOKEN",),
            public_api=True,
        )
