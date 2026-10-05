import io

from fastapi.testclient import TestClient
from PIL import Image

from services.gateway import main as gateway
from shared.contracts import ModelAPIError


client = TestClient(gateway.app)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _not_ready(base_url: str, *, request_id: str, timeout: float) -> None:
    raise ModelAPIError(
        503,
        "UPSTREAM_NOT_READY",
        "A required upstream model service is not ready.",
        details=[{"upstream": base_url, "reason": f"{base_url} refused the connection"}],
    )


def test_gateway_readiness_is_degraded_when_one_optional_pipeline_is_down(monkeypatch):
    monkeypatch.setenv("GATEWAY_ENABLED_PIPELINES", "ocr-custom,image-verification")
    monkeypatch.delenv("GATEWAY_REQUIRED_PIPELINES", raising=False)

    def probe(base_url: str, *, request_id: str, timeout: float) -> None:
        if base_url == gateway.IMAGE_VERIFICATION_URL:
            _not_ready(base_url, request_id=request_id, timeout=timeout)

    monkeypatch.setattr(gateway, "get_readiness", probe)
    response = client.get("/api/v1/readiness")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["status"] == "degraded"
    assert data["summary"] == {
        "enabled": 2,
        "ready": 1,
        "not_ready": 1,
        "required_not_ready": [],
    }
    statuses = {service["name"]: service["status"] for service in data["services"]}
    assert statuses == {"ocr-custom": "ready", "image-verification": "not_ready"}


def test_gateway_readiness_only_checks_enabled_pipelines(monkeypatch):
    monkeypatch.setenv("GATEWAY_ENABLED_PIPELINES", "ocr-custom")
    monkeypatch.delenv("GATEWAY_REQUIRED_PIPELINES", raising=False)
    checked = []

    def probe(base_url: str, *, request_id: str, timeout: float) -> None:
        checked.append(base_url)

    monkeypatch.setattr(gateway, "get_readiness", probe)
    response = client.get("/api/v1/readiness")

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "ready"
    assert checked == [gateway.OCR_CUSTOM_URL]


def test_gateway_readiness_checks_direct_leaf_dependencies(monkeypatch):
    monkeypatch.setenv(
        "GATEWAY_ENABLED_PIPELINES",
        "text-det-v5,text-det-v6,text-recognition,siglip",
    )
    monkeypatch.delenv("GATEWAY_REQUIRED_PIPELINES", raising=False)
    checked = []

    def probe(base_url: str, *, request_id: str, timeout: float) -> None:
        checked.append(base_url)

    monkeypatch.setattr(gateway, "get_readiness", probe)
    response = client.get("/api/v1/readiness")

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "ready"
    assert checked == [
        gateway.TEXT_DETECTION_URL,
        gateway.REC_SERVICE_URL,
        gateway.SIGLIP_URL,
    ]


def test_gateway_readiness_fails_when_a_required_pipeline_is_down(monkeypatch):
    monkeypatch.setenv("GATEWAY_ENABLED_PIPELINES", "ocr-custom,image-verification")
    monkeypatch.setenv("GATEWAY_REQUIRED_PIPELINES", "image-verification")

    def probe(base_url: str, *, request_id: str, timeout: float) -> None:
        if base_url == gateway.IMAGE_VERIFICATION_URL:
            _not_ready(base_url, request_id=request_id, timeout=timeout)

    monkeypatch.setattr(gateway, "get_readiness", probe)
    response = client.get("/api/v1/readiness")

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "REQUIRED_UPSTREAMS_NOT_READY"
    assert error["details"][0]["summary"]["required_not_ready"] == ["image-verification"]


def test_gateway_readiness_fails_when_every_enabled_pipeline_is_down(monkeypatch):
    monkeypatch.setenv("GATEWAY_ENABLED_PIPELINES", "ocr-custom,image-verification")
    monkeypatch.delenv("GATEWAY_REQUIRED_PIPELINES", raising=False)
    monkeypatch.setattr(gateway, "get_readiness", _not_ready)

    response = client.get("/api/v1/readiness")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "NO_UPSTREAMS_READY"


def test_text_detection_version_routes_directly_to_v5(monkeypatch):
    forwarded = {}

    async def forward(request, *, upstream, endpoint, extra_fields=None, field_resolver=None):
        forwarded.update(upstream=upstream, endpoint=endpoint)
        return {"selected": "v5"}

    monkeypatch.setattr(gateway, "_forward", forward)
    response = client.post("/api/v1/text-detections?version=v5")

    assert response.status_code == 200
    assert response.json() == {"selected": "v5"}
    assert forwarded == {
        "upstream": gateway.TEXT_DETECTION_URL,
        "endpoint": "/api/v1/text-detections",
    }


def test_text_detection_version_routes_directly_to_v6(monkeypatch):
    forwarded = {}

    async def forward(request, *, upstream, endpoint, extra_fields=None, field_resolver=None):
        forwarded.update(upstream=upstream, endpoint=endpoint)
        return {"selected": "v6"}

    monkeypatch.setattr(gateway, "_forward", forward)
    response = client.post("/api/v1/text-detections?version=v6")

    assert response.status_code == 200
    assert response.json() == {"selected": "v6"}
    assert forwarded == {
        "upstream": gateway.TEXT_DETECTION_URL,
        "endpoint": "/api/v1/text-detections",
    }


def test_text_detection_without_version_routes_directly_to_leaf(monkeypatch):
    forwarded = {}

    async def forward(request, *, upstream, endpoint, extra_fields=None, field_resolver=None):
        fields = field_resolver({}) if field_resolver is not None else (extra_fields or {})
        selected_upstream = upstream(fields) if callable(upstream) else upstream
        forwarded.update(upstream=selected_upstream, endpoint=endpoint)
        return {"selected": "configured-default"}

    monkeypatch.setattr(gateway, "_forward", forward)
    response = client.post("/api/v1/text-detections")

    assert response.status_code == 200
    assert response.json() == {"selected": "configured-default"}
    assert forwarded["upstream"] == gateway.TEXT_DETECTION_URL


def test_text_detection_rejects_unknown_version():
    response = client.post("/api/v1/text-detections?version=v7")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_MODEL_VERSION"


def test_text_detection_forwards_model_variant_and_numeric_version(monkeypatch):
    forwarded = {}

    async def forward(request, *, upstream, endpoint, extra_fields=None, field_resolver=None):
        forwarded.update(
            upstream=upstream,
            endpoint=endpoint,
            extra_fields=extra_fields,
        )
        return {"selected": "thai_ft_v1"}

    monkeypatch.setattr(gateway, "_forward", forward)
    response = client.post("/api/v1/text-detections?version=6&model=thai_ft_v1")

    assert response.status_code == 200
    assert forwarded == {
        "upstream": gateway.TEXT_DETECTION_URL,
        "endpoint": "/api/v1/text-detections",
        "extra_fields": {"version": "v6", "model": "thai_ft_v1"},
    }


def test_text_detection_uses_multipart_version_for_upstream_and_variant(monkeypatch):
    forwarded = {}

    def post_images(base_url, endpoint, image_paths, *, fields, request_id, multiple=False):
        forwarded.update(
            base_url=base_url,
            endpoint=endpoint,
            fields=fields,
            multiple=multiple,
        )
        return {"predictions": []}

    monkeypatch.setattr(gateway, "post_images", post_images)
    response = client.post(
        "/api/v1/text-detections",
        files={"image": ("sample.png", _png_bytes(), "image/png")},
        data={"version": "6", "model": "baseline"},
    )

    assert response.status_code == 200
    assert forwarded == {
        "base_url": gateway.TEXT_DETECTION_URL,
        "endpoint": "/api/v1/text-detections",
        "fields": {"version": "v6", "model": "baseline"},
        "multiple": False,
    }


def test_text_detection_variant_requires_version():
    response = client.post("/api/v1/text-detections?model=thai_ft_v1")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MODEL_VERSION_REQUIRED"


def test_text_recognition_forwards_version_and_model(monkeypatch):
    forwarded = {}

    async def forward(request, *, upstream, endpoint, extra_fields=None, field_resolver=None):
        forwarded.update(
            upstream=upstream,
            endpoint=endpoint,
            extra_fields=extra_fields,
        )
        return {"selected": "thai_ft_v2"}

    monkeypatch.setattr(gateway, "_forward", forward)
    response = client.post("/api/v1/text-recognitions?version=6&model=thai_ft_v2")

    assert response.status_code == 200
    assert forwarded == {
        "upstream": gateway.REC_SERVICE_URL,
        "endpoint": "/api/v1/text-recognitions",
        "extra_fields": {"version": "v6", "model": "thai_ft_v2"},
    }


def test_table_v2_forwards_ocr_version_and_profile(monkeypatch):
    forwarded = {}

    async def forward(request, *, upstream, endpoint, extra_fields=None, field_resolver=None):
        forwarded.update(
            upstream=upstream,
            endpoint=endpoint,
            extra_fields=extra_fields,
        )
        return {"selected": "thai_ft_v1"}

    monkeypatch.setattr(gateway, "_forward", forward)
    response = client.post(
        "/api/v1/table-model-results?ocr_version=5&profile=thai_ft_v1"
    )

    assert response.status_code == 200
    assert forwarded == {
        "upstream": gateway.TABLE_MODEL_URL,
        "endpoint": "/api/v1/table-model-results",
        "extra_fields": {"version": "v5", "model": "thai_ft_v1"},
    }


def test_table_v2_profile_requires_ocr_version():
    response = client.post(
        "/api/v1/table-model-results?profile=thai_ft_v1"
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MODEL_VERSION_REQUIRED"


def test_table_v2_forwards_independent_recognition_variant(monkeypatch):
    forwarded = {}

    async def forward(request, *, upstream, endpoint, extra_fields=None, field_resolver=None):
        forwarded.update(extra_fields=extra_fields)
        return {"selected": "rec-only"}

    monkeypatch.setattr(gateway, "_forward", forward)
    response = client.post(
        "/api/v1/table-model-results?ocr_version=5&rec_model=thai_ft_v1"
    )

    assert response.status_code == 200
    assert forwarded["extra_fields"] == {
        "version": "v5",
        "model": "baseline",
        "rec_model": "thai_ft_v1",
    }


def test_ocr_result_batches_route_to_integrated_paddle(monkeypatch):
    forwarded = {}

    async def forward_multiple(request, *, upstream, endpoint, field_resolver=None):
        forwarded.update(upstream=upstream, endpoint=endpoint)
        return {"selected": "paddle"}

    monkeypatch.setattr(gateway, "_forward_multiple", forward_multiple)
    response = client.post("/api/v1/ocr-result-batches?engine=paddle")

    assert response.status_code == 200
    assert response.json() == {"selected": "paddle"}
    assert forwarded == {
        "upstream": gateway.OCR_PADDLE_URL,
        "endpoint": "/api/v1/ocr-result-batches",
    }


def test_custom_ocr_forwards_independent_det_and_rec_variants(monkeypatch):
    forwarded = {}

    async def forward(request, *, upstream, endpoint, extra_fields=None, field_resolver=None):
        forwarded.update(
            upstream=upstream,
            endpoint=endpoint,
            extra_fields=extra_fields,
        )
        return {"selected": "custom"}

    monkeypatch.setattr(gateway, "_forward", forward)
    response = client.post(
        "/api/v1/ocr-results?engine=custom&version=6&model=baseline"
        "&det_model=thai_ft_v1&rec_model=thai_ft_v2"
    )

    assert response.status_code == 200
    assert forwarded == {
        "upstream": gateway.OCR_CUSTOM_URL,
        "endpoint": "/api/v1/ocr-results",
        "extra_fields": {
            "version": "v6",
            "model": "baseline",
            "det_model": "thai_ft_v1",
            "rec_model": "thai_ft_v2",
        },
    }


def test_custom_ocr_variant_requires_version():
    response = client.post(
        "/api/v1/ocr-results?engine=custom&rec_model=thai_ft_v1"
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MODEL_VERSION_REQUIRED"


def test_ocr_result_batches_reject_custom_engine():
    response = client.post("/api/v1/ocr-result-batches?engine=custom")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_image_classifications_route_forwards_directly_to_siglip(monkeypatch):
    forwarded = {}

    async def forward(request, *, upstream, endpoint, extra_fields=None, field_resolver=None):
        forwarded.update(upstream=upstream, endpoint=endpoint)
        return {"logits": [2.0, 0.0, -1.0], "device": "cuda:0"}

    monkeypatch.setattr(gateway, "_forward", forward)
    response = client.post("/api/v1/image-classifications")

    assert response.status_code == 200
    assert response.json()["logits"] == [2.0, 0.0, -1.0]
    assert forwarded == {
        "upstream": gateway.SIGLIP_URL,
        "endpoint": "/api/v1/image-classifications",
    }
