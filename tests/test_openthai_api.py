import base64
import io
import threading
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from inference.openthai.settings import Settings
from services.ocr_openthai.main import ENDPOINT, create_app
from services.gateway import main as gateway

class Reader:
    def read(self, image):
        return {"text": "hello", "lines": [
            {"text": "hello", "confidence": 0.9, "row": 0,
             "box": [[0, 0], [3, 0], [3, 3], [0, 3]]}]}

def png():
    out = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(out, format="PNG")
    return out.getvalue()

@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-openthai-internal")
    monkeypatch.setenv("INTERNAL_API_TOKEN_PREVIOUS", "")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10000")

@pytest.mark.parametrize("multipart", [True, False])
def test_success(multipart):
    with TestClient(create_app(Settings(), loader=Reader), headers={"X-API-Key": "test-openthai-internal"}) as client:
        kwargs = ({"files": {"image": ("a.png", png(), "image/png")}} if multipart
                  else {"json": {"image": base64.b64encode(png()).decode()}})
        response = client.post(ENDPOINT, headers={"X-Request-ID": "req_openthai"}, **kwargs)
        assert response.status_code == 200
        body = response.json()
        assert body["data"]["text"] == "hello"
        assert body["data"]["image"] == {"width": 4, "height": 4}
        assert body["meta"]["request_id"] == "req_openthai"
        assert body["meta"]["service"] == "leaf-openthai-ocr"
        assert client.get("/paddle/api/v1/readiness").status_code == 200

def test_load_failure():
    def fail():
        raise ValueError("checksum mismatch")
    with TestClient(create_app(Settings(), loader=fail), headers={"X-API-Key": "test-openthai-internal"}) as client:
        assert client.get("/paddle/api/v1/health").status_code == 200
        assert client.get("/paddle/api/v1/readiness").status_code == 503
        assert client.post(ENDPOINT).json()["error"]["code"] == "MODEL_NOT_READY"

def test_timeout_does_not_release_native_job():
    release = threading.Event()
    started = threading.Event()
    class SlowReader(Reader):
        def read(self, image):
            started.set()
            release.wait(5)
            return super().read(image)
    app = create_app(Settings(request_timeout_seconds=0.1), loader=SlowReader)
    with TestClient(app, headers={"X-API-Key": "test-openthai-internal"}) as client:
        try:
            first = client.post(ENDPOINT, files={"image": ("a.png", png(), "image/png")})
            assert started.is_set()
            assert first.status_code == 504
            assert not app.state.inference.pending.done()
            second = client.post(ENDPOINT, files={"image": ("a.png", png(), "image/png")})
            assert second.status_code == 503
            assert second.json()["error"]["code"] == "SERVICE_BUSY"
        finally:
            release.set()
        app.state.inference.pending.result(timeout=5)

def test_gateway_forwards_and_wraps_once(monkeypatch):
    def post(base_url, endpoint, paths, **kwargs):
        assert base_url == gateway.OCR_OPENTHAI_URL
        assert endpoint == ENDPOINT
        assert paths[0].exists()
        assert kwargs["request_id"] == "req_forward"
        return {"text": "hello", "lines": []}
    monkeypatch.setattr(gateway, "post_images", post)
    with TestClient(gateway.app) as client:
        response = client.post(ENDPOINT, files={"image": ("a.png", png(), "image/png")},
                               headers={"X-Request-ID": "req_forward"})
    assert response.status_code == 200
    assert response.json()["data"] == {"text": "hello", "lines": []}
    assert response.json()["meta"]["service"] == gateway.SERVICE_NAME

def test_limits_and_bad_model_output():
    with TestClient(create_app(Settings(max_pixels=1), loader=Reader), headers={"X-API-Key": "test-openthai-internal"}) as client:
        response = client.post(ENDPOINT, files={"image": ("a.png", png(), "image/png")})
        assert response.status_code == 413
    class BadReader:
        def read(self, image):
            return {"text": "bad", "lines": [{"confidence": float("nan")}]}
    with TestClient(create_app(Settings(), loader=BadReader), headers={"X-API-Key": "test-openthai-internal"}) as client:
        response = client.post(ENDPOINT, files={"image": ("a.png", png(), "image/png")})
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "MODEL_INFERENCE_FAILED"

def test_internal_auth(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-internal-token")
    with TestClient(create_app(Settings(), loader=Reader)) as client:
        assert client.post(ENDPOINT).status_code == 401
        response = client.post(ENDPOINT, files={"image": ("a.png", png(), "image/png")},
                               headers={"Authorization": "Bearer test-internal-token"})
        assert response.status_code == 200


def test_output_size_limit():
    class LargeReader:
        def read(self, image):
            return {"text": "x" * 2000, "lines": []}
    with TestClient(create_app(Settings(max_response_bytes=1024), loader=LargeReader), headers={"X-API-Key": "test-openthai-internal"}) as client:
        response = client.post(ENDPOINT, files={"image": ("a.png", png(), "image/png")})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "OUTPUT_TOO_LARGE"


def test_checksum_rejected_before_import(tmp_path):
    from inference.openthai.engine import Inference
    (tmp_path / "det.onnx").write_bytes(b"invalid")
    inference = Inference(Settings(model_dir=tmp_path))
    try:
        with pytest.raises(ValueError, match="checksum mismatch"):
            inference.load()
        assert inference.reader is None
    finally:
        inference.executor.shutdown()
