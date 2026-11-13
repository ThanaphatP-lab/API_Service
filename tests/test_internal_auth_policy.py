import pytest
from fastapi.testclient import TestClient
from core.app_factory import create_app

INTERNAL = "internal-test-secret"
PREVIOUS = "previous-internal-test-secret"
GATEWAY = "gateway-test-secret"


def app():
    service = create_app("Internal test", "fake")
    @service.post("/infer")
    def infer():
        return {"ok": True}
    @service.get("/paddle/api/v1/readiness")
    def readiness():
        return {"status": "ready"}
    return service


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("INTERNAL_API_TOKEN", INTERNAL)
    monkeypatch.setenv("INTERNAL_API_TOKEN_PREVIOUS", PREVIOUS)
    monkeypatch.setenv("MODEL_GATEWAY_API_KEY", GATEWAY)
    monkeypatch.setenv("OPENTHAI_API_KEY", "old-openthai-secret")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10000")


@pytest.mark.parametrize("header", ["Authorization", "X-API-Key"])
@pytest.mark.parametrize("token,status", [
    (INTERNAL, 200), (PREVIOUS, 200), (GATEWAY, 401),
    ("old-openthai-secret", 401), ("wrong", 401),
])
def test_only_internal_credentials(header, token, status):
    value = "Bearer " + token if header == "Authorization" else token
    client = TestClient(app())
    assert client.post("/infer", headers={header: value}).status_code == status
    assert client.get("/paddle/api/v1/readiness", headers={header: value}).status_code == status


def test_missing_client_key():
    assert TestClient(app()).post("/infer").status_code == 401


def test_missing_server_key_is_not_anonymous(monkeypatch):
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)
    client = TestClient(app())
    for headers in ({}, {"X-API-Key": GATEWAY}, {"X-API-Key": PREVIOUS}):
        response = client.post("/infer", headers=headers)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "AUTH_CONFIGURATION_ERROR"
    assert client.get("/paddle/api/v1/readiness").status_code == 503
    assert client.get("/health").status_code == 200


def test_internal_factory_rejects_other_key():
    with pytest.raises(ValueError, match="INTERNAL_API_TOKEN"):
        create_app("Internal", "fake", auth_token_env="OPENTHAI_API_KEY")


def test_gateway_uses_its_own_key():
    service = create_app("Gateway", "fake", public_api=True,
                         auth_token_env="MODEL_GATEWAY_API_KEY")
    @service.post("/infer")
    def infer():
        return {"ok": True}
    client = TestClient(service)
    assert client.post("/infer", headers={"X-API-Key": GATEWAY}).status_code == 200
    assert client.post("/infer", headers={"X-API-Key": INTERNAL}).status_code == 401
