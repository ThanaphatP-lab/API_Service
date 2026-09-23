import json
from pathlib import Path

from fastapi.testclient import TestClient

from services.gateway import main as gateway


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "API_V1_CONTRACT.json").read_text(encoding="utf-8"))


def test_gateway_openapi_matches_frozen_public_endpoint_inventory():
    schema = gateway.app.openapi()
    actual = {
        (method.upper(), path)
        for path, operations in schema["paths"].items()
        for method in operations
        if method.lower() in {"get", "post", "put", "patch", "delete"}
    }
    expected = {
        (entry["method"], entry["path"])
        for entry in CONTRACT["public_endpoints"]
    }

    assert actual == expected


def test_hidden_legacy_routes_remain_outside_openapi():
    schema_paths = gateway.app.openapi()["paths"]

    for endpoint in CONTRACT["hidden_legacy_endpoints"]:
        assert endpoint["path"] not in schema_paths


def test_success_envelope_matches_frozen_contract():
    response = TestClient(gateway.app).get(
        "/api/v1/health",
        headers={"X-Request-ID": "req_phase0_contract"},
    )

    assert response.status_code == 200
    body = response.json()
    envelope = CONTRACT["success_envelope"]
    assert set(envelope["required"]) <= set(body)
    assert set(envelope["meta_required"]) <= set(body["meta"])
    assert body["meta"]["request_id"] == "req_phase0_contract"
    assert body["meta"]["api_version"] == "v1"


def test_error_envelope_matches_frozen_contract():
    response = TestClient(gateway.app).post(
        "/api/v1/text-recognitions",
        json={},
        headers={"X-Request-ID": "req_phase0_error"},
    )

    assert response.status_code == 422
    body = response.json()
    envelope = CONTRACT["error_envelope"]
    assert set(envelope["required"]) <= set(body)
    assert set(envelope["error_required"]) <= set(body["error"])
    assert body["error"]["request_id"] == "req_phase0_error"
    assert body["error"]["code"] == "IMAGE_REQUIRED"
