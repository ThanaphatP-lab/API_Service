import io
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from core.app_factory import create_app
from pipelines.layout.contracts import format_legacy_detection
from services.gateway import main as gateway


@pytest.mark.parametrize("path", ["/predict", "/v1/textdetection", "/v1/textrecognition"])
def test_legacy_flag_blocks_and_reenables_alias(monkeypatch, path):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "0")
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)
    app = create_app("test", "fake")
    app.add_api_route(path, lambda: {"ok": True}, methods=["POST"])
    client = TestClient(app)
    monkeypatch.setenv("LEGACY_ENDPOINTS_ENABLED", "false")
    assert client.post(path).status_code == 404
    assert client.get("/health").status_code == 200
    monkeypatch.setenv("LEGACY_ENDPOINTS_ENABLED", "true")
    assert client.post(path).status_code == 200


@pytest.mark.parametrize("batch", [False, True])
def test_direct_leaf_legacy_format_matches_previous_pipeline(tmp_path, batch):
    paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for path in paths:
        Image.new("RGB", (40, 30), "white").save(path)
    prediction = {"dt_polys": [[[1, 2], [20, 2], [20, 12], [1, 12]]], "dt_scores": [0.9]}
    data = {"predictions": [prediction]}  # missing second result intentionally

    result = format_legacy_detection(data, paths if batch else paths[:1], multiple=batch)
    first = result["results"][0] if batch else result
    assert first["image_width"] == 40
    assert first["image_height"] == 30
    assert first["raw_predictions"] == [prediction]
    assert first["regions"] == [{
        "text": "", "confidence": 0.9,
        "bbox": {"x": 1.0, "y": 2.0, "width": 19.0, "height": 10.0,
                 "x_ratio": 0.025, "y_ratio": 0.066667,
                 "width_ratio": 0.475, "height_ratio": 0.333333},
        "polygon": [[1.0, 2.0], [20.0, 2.0], [20.0, 12.0], [1.0, 12.0]],
        "type": "text", "data_type": "text", "label": "text", "source": "text-detection",
    }]
    if batch:
        assert result["count"] == 2
        assert result["results"][1]["regions"] == []


@pytest.mark.parametrize("batch", [False, True])
def test_unversioned_gateway_calls_only_det_with_compat_format(monkeypatch, batch):
    calls = []
    def post(url, endpoint, paths, **kwargs):
        calls.append((url, kwargs["fields"]))
        return {"regions": []}
    monkeypatch.setattr(gateway, "post_images", post)
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    route = "/api/v1/text-detection-batches" if batch else "/api/v1/text-detections"
    response = TestClient(gateway.app).post(route, files={
        "images" if batch else "image": ("a.png", buffer.getvalue(), "image/png"),
    })
    assert response.status_code == 200
    assert calls == [(gateway.TEXT_DETECTION_URL, {"model": "baseline", "response_contract": "legacy-layout"})]


def test_retired_local_code_is_absent():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "models/version_old/det_old.py").exists()
    assert not (root / "models/version_old/rec.py").exists()
    from pipelines.ocr import orchestrator
    assert not hasattr(orchestrator, "predict_custom_ocr")
    for relative in (
        "shared/api.py", "pipeline/ocr_pipeline.py", "pipeline/layout_pipeline.py",
        "pipeline/table_pipeline.py", "pipeline/image_verification_pipeline.py",
        "models/det_model.py", "models/rec_model.py", "models/table_v2_model.py",
        "services/ocr_pipeline_custom/compatibility.py",
    ):
        assert not (root / relative).exists()


def test_layout_keeps_document_workflow_but_no_det_only_routes():
    from services.layout_pipeline import main as layout
    paths = layout.app.openapi()["paths"]
    assert "/api/v1/document-layouts" in paths
    assert "/api/v1/text-detections" not in paths
    assert "/api/v1/text-detection-batches" not in paths


def test_recognition_only_routes_cannot_be_reenabled(monkeypatch):
    from services.ocr_pipeline_custom import main as custom
    monkeypatch.setenv("OCR_CUSTOM_RECOGNITION_COMPAT_ENABLED", "true")
    for path in ("/api/v1/text-recognitions", "/api/v1/text-recognition-batches"):
        assert TestClient(custom.app).post(path).status_code == 404
    assert "/api/v1/ocr-results" in custom.app.openapi()["paths"]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_leaf_http_formats_legacy_only_on_request(monkeypatch, batch, legacy):
    root = Path(__file__).resolve().parents[1]
    fake = types.ModuleType("inference.text_detection")
    prediction = {"dt_polys": [], "dt_scores": []}
    fake.get_model = lambda *args: None
    fake.infer = lambda *args: {"predictions": [prediction], **prediction}
    fake.infer_batch = lambda *args: {"predictions": [prediction], "results": [prediction]}
    monkeypatch.setitem(sys.modules, "inference.text_detection", fake)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "0")
    spec = importlib.util.spec_from_file_location("det_http_test", root / "services/text_det/main.py")
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    buffer = io.BytesIO()
    Image.new("RGB", (4, 6), "white").save(buffer, format="PNG")
    endpoint = "/api/v1/text-detection-batches" if batch else "/api/v1/text-detections"
    response = TestClient(service.app).post(endpoint, files={
        "images" if batch else "image": ("a.png", buffer.getvalue(), "image/png"),
    }, data={"response_contract": "legacy-layout" if legacy else "leaf", "version": "6"})
    assert response.status_code == 200
    data = response.json()["data"]
    item = data["results"][0] if batch else data
    if legacy:
        assert item["regions"] == []
        assert (item["image_width"], item["image_height"]) == (4, 6)
    else:
        assert item["dt_polys"] == []
    assert response.json()["meta"]["service"] == "leaf-text-detection"
