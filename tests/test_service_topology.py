import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from core.topology import configured_services
from services.gateway import main as gateway
from services.ocr_pipeline_custom import main as custom


@pytest.fixture(autouse=True)
def clean_topology(monkeypatch):
    for key in ("GATEWAY_ENABLED_SERVICES", "GATEWAY_REQUIRED_SERVICES",
                "GATEWAY_ENABLED_PIPELINES", "GATEWAY_REQUIRED_PIPELINES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(gateway, "TEXT_DETECTION_URL", "http://localhost:8002")


def png():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("version", ["5", "6"])
@pytest.mark.parametrize("batch", [False, True])
def test_unified_detection_routes_single_and_batch_with_selection(monkeypatch, version, batch):
    monkeypatch.setattr(gateway, "TEXT_DETECTION_URL", "http://unified-det")
    calls = []

    def post(url, endpoint, paths, **kwargs):
        calls.append((url, endpoint, len(paths), kwargs))
        return {"predictions": []}

    monkeypatch.setattr(gateway, "post_images", post)
    endpoint = "/api/v1/text-detection-batches" if batch else "/api/v1/text-detections"
    response = TestClient(gateway.app).post(
        endpoint, files={"images" if batch else "image": ("test.png", png(), "image/png")},
        data={"version": version, "model": "thai_ft_v1"},
    )
    assert response.status_code == 200
    assert calls[0][:3] == ("http://unified-det", endpoint, 1)
    assert calls[0][3]["fields"] == {"version": "v" + version, "model": "thai_ft_v1"}
    assert calls[0][3].get("multiple", False) is batch


def test_unified_detection_retains_legacy_unversioned_contract_and_validation(monkeypatch):
    monkeypatch.setattr(gateway, "TEXT_DETECTION_URL", "http://unified-det")
    assert gateway._text_detector_upstream(None) == "http://unified-det"
    response = TestClient(gateway.app).post("/api/v1/text-detections?version=7")
    assert response.status_code == 422


def test_both_versions_share_default_detector():
    assert gateway._text_detector_upstream("5") == gateway.TEXT_DETECTION_URL
    assert gateway._text_detector_upstream("6") == gateway.TEXT_DETECTION_URL


def test_unified_readiness_aliases_probe_one_service(monkeypatch):
    monkeypatch.setattr(gateway, "TEXT_DETECTION_URL", "http://unified-det")
    monkeypatch.setenv("GATEWAY_ENABLED_PIPELINES", "text-det-v5,text-det-v6")
    monkeypatch.setenv("GATEWAY_REQUIRED_PIPELINES", "text-det-v6")
    calls = []
    monkeypatch.setattr(gateway, "get_readiness", lambda url, **kwargs: calls.append(url))
    response = TestClient(gateway.app).get("/api/v1/readiness")
    assert response.status_code == 200
    assert calls == ["http://unified-det"]
    data = response.json()["data"]
    assert data["summary"]["enabled"] == 1
    assert data["services"][0]["name"] == "text-detection"
    assert data["services"][0]["required"] is True
    assert data["services"][0]["kind"] == "leaf"


def test_discovery_only_advertises_configured_capabilities_without_probes(monkeypatch):
    monkeypatch.setenv("GATEWAY_ENABLED_SERVICES", "leaf-text-recognition,pipeline-table-v2")
    monkeypatch.setenv("GATEWAY_ENABLED_PIPELINES", "all")
    monkeypatch.setenv("GATEWAY_REQUIRED_SERVICES", "leaf-text-recognition")

    def unexpected(*args, **kwargs):
        pytest.fail("discovery must not load models or probe HTTP")

    monkeypatch.setattr(gateway, "get_readiness", unexpected)
    response = TestClient(gateway.app).get("/api/v1/services")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["document_layouts"] == "/api/v1/document-layouts"  # old inventory retained
    capabilities = {item["name"]: item for item in data["capabilities"]}
    assert set(capabilities) == {"text-recognition", "table-model"}
    assert capabilities["text-recognition"]["required"] is True
    assert capabilities["table-model"]["kind"] == "pipeline"
    assert "http://" not in str(capabilities)


@pytest.mark.parametrize("enabled,required", [("unknown", ""), ("siglip", "text-recognition"), ("", "")])
def test_invalid_configuration_matches_in_discovery_and_readiness(monkeypatch, enabled, required):
    monkeypatch.setenv("GATEWAY_ENABLED_SERVICES", enabled)
    monkeypatch.setenv("GATEWAY_REQUIRED_SERVICES", required)
    for path in ("/api/v1/services", "/api/v1/readiness"):
        response = TestClient(gateway.app).get(path)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "GATEWAY_CONFIGURATION_ERROR"


def test_legacy_recognition_routes_are_disabled_by_default():
    schema = custom.app.openapi()
    for path in ("/api/v1/text-recognitions", "/api/v1/text-recognition-batches"):
        assert path not in schema["paths"]
