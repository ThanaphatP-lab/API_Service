from __future__ import annotations

import asyncio
import logging
from core.service_settings import ServiceURLs
from core.settings import runtime_settings
from core.topology import SERVICE_DESCRIPTORS, capabilities, configured_services, validate_services
import time
from collections.abc import Callable
from typing import Any

from fastapi import Request

from core.request_parsing import BATCH_IMAGE_REQUEST_OPENAPI, IMAGE_REQUEST_OPENAPI, parse_image_request
from core.app_factory import create_app
from shared.contracts import ModelAPIError, request_id, success_response
from shared.model_variants import normalize_model_variant, normalize_model_version
from shared.upstream import get_readiness, post_images


logger = logging.getLogger("uvicorn.error")
SERVICE_NAME = "model-api-gateway"
MODEL_NAME = "pipeline-router-v1"
service_urls = ServiceURLs()
TEXT_DETECTION_URL = service_urls.text_detection_url
LAYOUT_PIPELINE_URL = service_urls.layout_pipeline_url
REC_SERVICE_URL = service_urls.rec_service_url
OCR_CUSTOM_URL = service_urls.ocr_custom_url
OCR_PADDLE_URL = service_urls.ocr_paddle_url
TABLE_PIPELINE_URL = service_urls.table_pipeline_url
TABLE_MODEL_URL = service_urls.table_model_url
SIGLIP_URL = service_urls.siglip_url
LAYOUT_SERVICE_URL = service_urls.layout_service_url
IMAGE_VERIFICATION_URL = service_urls.image_verification_url
app = create_app(
    "Model API Gateway",
    MODEL_NAME,
    service_name=SERVICE_NAME,
    auth_token_env="MODEL_GATEWAY_API_KEY",
    required_auth_token_envs=("INTERNAL_API_TOKEN",),
    public_api=True,
)

UPSTREAMS = {
    "layout": LAYOUT_PIPELINE_URL,
    "layout-model": LAYOUT_SERVICE_URL,
    "ocr-custom": OCR_CUSTOM_URL,
    "ocr-paddle": OCR_PADDLE_URL,
    "table": TABLE_PIPELINE_URL,
    "table-model": TABLE_MODEL_URL,
    "image-verification": IMAGE_VERIFICATION_URL,
    "text-detection": TEXT_DETECTION_URL,
    "text-recognition": REC_SERVICE_URL,
    "siglip": SIGLIP_URL,
}


def _pipeline_names(variable: str, *, default_all: bool = False) -> set[str]:
    return configured_services(variable, _active_upstreams(), default_all=default_all)


def _active_upstreams() -> dict[str, str]:
    return {**UPSTREAMS, "text-detection": TEXT_DETECTION_URL or service_urls.text_detection_url}


def _readiness_timeout() -> float:
    return runtime_settings.gateway_readiness_timeout


def _text_detector_upstream(version: Any) -> str:
    if version is not None and str(version).strip():
        normalize_model_version(version)
    return TEXT_DETECTION_URL or service_urls.text_detection_url


def _model_selection_fields(
    version: str | None,
    model: str | None,
    *,
    require_version_for_variant: bool = False,
) -> dict[str, str]:
    selected_model = normalize_model_variant(model)
    if require_version_for_variant and not version and selected_model != "baseline":
        raise ModelAPIError(
            422,
            "MODEL_VERSION_REQUIRED",
            "version is required when selecting a non-baseline model.",
            details=[{"field": "version", "model": selected_model}],
        )
    fields = {"model": selected_model}
    if version:
        fields["version"] = normalize_model_version(version)
    return fields


def _table_v2_selection_fields(
    version: str | None,
    model: str | None,
    det_model: str | None,
    rec_model: str | None,
) -> dict[str, str]:
    fields = _model_selection_fields(
        version,
        model,
        require_version_for_variant=True,
    )
    if det_model is not None:
        fields["det_model"] = normalize_model_variant(det_model)
    if rec_model is not None:
        fields["rec_model"] = normalize_model_variant(rec_model)
    if not version and any(
        fields.get(name, "baseline") != "baseline"
        for name in ("model", "det_model", "rec_model")
    ):
        raise ModelAPIError(
            422,
            "MODEL_VERSION_REQUIRED",
            "version is required when selecting a non-baseline model.",
            details=[{"field": "version"}],
        )
    return fields


def _field_value(request: Request, fields: dict[str, Any], name: str, default: Any = None) -> Any:
    if name in request.query_params:
        return request.query_params[name]
    return fields.get(name, default)


def _request_model_selection(
    request: Request,
    fields: dict[str, Any],
    *,
    require_version_for_variant: bool = False,
) -> dict[str, str]:
    return _model_selection_fields(
        _field_value(request, fields, "version"),
        _field_value(request, fields, "model", "baseline"),
        require_version_for_variant=require_version_for_variant,
    )


def _request_ocr_selection(request: Request, fields: dict[str, Any]) -> dict[str, str]:
    return _table_v2_selection_fields(
        _field_value(request, fields, "version"),
        _field_value(request, fields, "model", "baseline"),
        _field_value(request, fields, "det_model"),
        _field_value(request, fields, "rec_model"),
    )


def _request_paddle_selection(request: Request, fields: dict[str, Any]) -> dict[str, str]:
    common = _field_value(request, fields, "ocr_version", _field_value(request, fields, "version"))
    model = normalize_model_variant(_field_value(request, fields, "profile", _field_value(request, fields, "model", "baseline")))
    selected = {"model": model}
    if common is not None:
        selected["version"] = normalize_model_version(common)
    for prefix in ("det", "rec"):
        version = _field_value(request, fields, f"{prefix}_version")
        variant = normalize_model_variant(_field_value(request, fields, f"{prefix}_model", model))
        if version is not None:
            selected[f"{prefix}_version"] = normalize_model_version(version)
        selected[f"{prefix}_model"] = variant
        if variant != "baseline" and version is None and common is None:
            raise ModelAPIError(422, "MODEL_VERSION_REQUIRED",
                                f"version or {prefix}_version is required for a non-baseline {prefix} model.")
    # Do not inject baseline per-module selectors into a legacy default request.
    for prefix in ("det", "rec"):
        if _field_value(request, fields, f"{prefix}_model") is None:
            selected.pop(f"{prefix}_model")
    return selected


def _request_table_v2_selection(request: Request, fields: dict[str, Any]) -> dict[str, str]:
    selected_version = _field_value(
        request,
        fields,
        "ocr_version",
        _field_value(request, fields, "version"),
    )
    selected_profile = _field_value(
        request,
        fields,
        "profile",
        _field_value(request, fields, "model", "baseline"),
    )
    return _table_v2_selection_fields(
        selected_version,
        selected_profile,
        _field_value(request, fields, "det_model"),
        _field_value(request, fields, "rec_model"),
    )


async def _probe_pipeline(name: str, url: str, *, current_request_id: str, timeout: float) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        await asyncio.to_thread(get_readiness, url, request_id=current_request_id, timeout=timeout)
        return {
            "name": name,
            "status": "ready",
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
    except ModelAPIError as exc:
        return {
            "name": name,
            "status": "not_ready",
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "error": {"code": exc.code, "message": exc.message},
        }
    except Exception as exc:  # Defensive: a readiness endpoint must still return a useful report.
        return {
            "name": name,
            "status": "not_ready",
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "error": {"code": "READINESS_CHECK_FAILED", "message": "The readiness check failed."},
        }


async def _forward(
    request: Request,
    *,
    upstream: str | Callable[[dict[str, Any]], str],
    endpoint: str,
    extra_fields: dict[str, Any] | None = None,
    field_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    image = await parse_image_request(request)
    try:
        resolved_fields = field_resolver(image.fields) if field_resolver is not None else {}
        fields = {**image.fields, **resolved_fields, **(extra_fields or {})}
        selected_upstream = upstream(fields) if callable(upstream) else upstream
        if endpoint == "/api/v1/text-detections":
            fields.pop("response_contract", None)
            if not fields.get("version"):
                fields["response_contract"] = "legacy-layout"
        data = await asyncio.to_thread(
            post_images,
            selected_upstream,
            endpoint,
            [image.path],
            fields=fields,
            request_id=request_id(request),
        )
        return success_response(request, data, service=SERVICE_NAME, model=MODEL_NAME)
    finally:
        image.cleanup()


async def _forward_multiple(
    request: Request,
    *,
    upstream: str | Callable[[dict[str, Any]], str],
    endpoint: str,
    extra_fields: dict[str, Any] | None = None,
    field_resolver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    images = await parse_image_request(request, multiple=True)
    try:
        resolved_fields = field_resolver(images.fields) if field_resolver is not None else {}
        fields = {**images.fields, **resolved_fields, **(extra_fields or {})}
        selected_upstream = upstream(fields) if callable(upstream) else upstream
        if endpoint == "/api/v1/text-detection-batches":
            fields.pop("response_contract", None)
            if not fields.get("version"):
                fields["response_contract"] = "legacy-layout"
        data = await asyncio.to_thread(
            post_images,
            selected_upstream,
            endpoint,
            images.paths,
            fields=fields,
            request_id=request_id(request),
            multiple=True,
        )
        return success_response(request, data, service=SERVICE_NAME, model=MODEL_NAME)
    finally:
        images.cleanup()


@app.get("/api/v1/services", tags=["Discovery"])
def services(request: Request) -> dict[str, Any]:
    available = _active_upstreams()
    enabled = _pipeline_names("GATEWAY_ENABLED_PIPELINES", default_all=True)
    required = _pipeline_names("GATEWAY_REQUIRED_PIPELINES")
    return success_response(
        request,
        {
            "document_layouts": "/api/v1/document-layouts",
            "document_layout_batches": "/api/v1/document-layout-batches",
            "layout_prediction_batches": "/api/v1/layout-prediction-batches",
            "image_classification_batches": "/api/v1/image-classification-batches",
            "ocr_results": "/api/v1/ocr-results",
            "ocr_result_batches": "/api/v1/ocr-result-batches",
            "text_detections": "/api/v1/text-detections",
            "text_detections_v5": "/api/v1/text-detections?version=v5",
            "text_detections_v6": "/api/v1/text-detections?version=v6",
            "text_detection_batches": "/api/v1/text-detection-batches",
            "text_recognitions": "/api/v1/text-recognitions",
            "text_recognition_batches": "/api/v1/text-recognition-batches",
            "table_results": "/api/v1/table-results",
            "table_model_results": "/api/v1/table-model-results",
            "image_classifications": "/api/v1/image-classifications",
            "image_verifications": "/api/v1/image-verifications",
            "capabilities": capabilities(available, enabled, required),
            "capability_status": "configured; use /api/v1/readiness for health",
            "detection_topology": "unified" if TEXT_DETECTION_URL else "split",
            "compatibility": {
                "unversioned_detection": "direct DET leaf with legacy response formatting",
                "route_map": "static API inventory; capabilities lists configured services",
            },
        },
        service=SERVICE_NAME,
        model=MODEL_NAME,
    )


@app.post("/api/v1/document-layouts", tags=["Public pipelines"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def document_layouts(request: Request) -> dict[str, Any]:
    return await _forward(
        request,
        upstream=LAYOUT_PIPELINE_URL,
        endpoint="/api/v1/document-layouts",
    )


@app.post("/api/v1/document-layout-batches", tags=["Public pipelines"], openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def document_layout_batches(request: Request) -> dict[str, Any]:
    return await _forward_multiple(request, upstream=LAYOUT_PIPELINE_URL,
                                   endpoint="/api/v1/document-layout-batches")


@app.post("/api/v1/layout-prediction-batches", tags=["Public model inference"], openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def layout_prediction_batches(request: Request) -> dict[str, Any]:
    return await _forward_multiple(request, upstream=LAYOUT_SERVICE_URL,
                                   endpoint="/api/v1/layout-prediction-batches")


@app.post("/api/v1/image-classification-batches", tags=["Public model inference"], openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def image_classification_batches(request: Request) -> dict[str, Any]:
    return await _forward_multiple(request, upstream=SIGLIP_URL,
                                   endpoint="/api/v1/image-classification-batches")


@app.post("/api/v1/ocr-results", tags=["Public pipelines"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def ocr_results(
    request: Request,
    engine: str = "custom",
    version: str | None = None,
    model: str = "baseline",
    det_model: str | None = None,
    rec_model: str | None = None,
) -> dict[str, Any]:
    engine = engine.strip().lower()
    if engine not in {"custom", "paddle"}:
        raise ModelAPIError(
            422,
            "VALIDATION_ERROR",
            "engine must be custom or paddle.",
            details=[{"field": "engine", "received": engine}],
        )
    upstream = OCR_CUSTOM_URL if engine == "custom" else OCR_PADDLE_URL
    selection_keys = {"version", "model", "det_model", "rec_model"}
    selection: dict[str, str] | None = None
    resolver = None
    if engine == "paddle":
        resolver = lambda fields: _request_paddle_selection(request, fields)
    elif selection_keys.intersection(request.query_params):
        selection = _table_v2_selection_fields(
            version,
            model,
            det_model,
            rec_model,
        )
    else:
        resolver = lambda fields: _request_ocr_selection(request, fields)
    return await _forward(
        request,
        upstream=upstream,
        endpoint="/api/v1/ocr-results",
        extra_fields=selection,
        field_resolver=resolver,
    )


@app.post("/api/v1/ocr-result-batches", tags=["Public pipelines"], openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def ocr_result_batches(request: Request) -> dict[str, Any]:
    """Run integrated PaddleOCR for a batch of backend ROI images."""
    engine = request.query_params.get("engine", "paddle").strip().lower()
    if engine != "paddle":
        raise ModelAPIError(
            422,
            "VALIDATION_ERROR",
            "batch OCR currently supports engine=paddle only.",
            details=[{"field": "engine", "received": engine}],
        )
    return await _forward_multiple(
        request,
        upstream=OCR_PADDLE_URL,
        endpoint="/api/v1/ocr-result-batches",
        field_resolver=lambda fields: _request_paddle_selection(request, fields),
    )


@app.post("/api/v1/text-detections", tags=["Public pipelines"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/v1/textdetection", include_in_schema=False)
async def text_detections(
    request: Request,
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    selection = _model_selection_fields(
        version,
        model,
        require_version_for_variant=True,
    )
    query_selects_model = any(name in request.query_params for name in ("version", "model"))
    return await _forward(
        request,
        upstream=(
            _text_detector_upstream(version)
            if query_selects_model
            else lambda fields: _text_detector_upstream(fields.get("version"))
        ),
        endpoint="/api/v1/text-detections",
        extra_fields=selection if query_selects_model else None,
        field_resolver=(
            None
            if query_selects_model
            else lambda fields: _request_model_selection(
                request,
                fields,
                require_version_for_variant=True,
            )
        ),
    )


@app.post("/api/v1/text-detection-batches", tags=["Public pipelines"], openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def text_detection_batches(
    request: Request,
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    selection = _model_selection_fields(
        version,
        model,
        require_version_for_variant=True,
    )
    query_selects_model = any(name in request.query_params for name in ("version", "model"))
    return await _forward_multiple(
        request,
        upstream=(
            _text_detector_upstream(version)
            if query_selects_model
            else lambda fields: _text_detector_upstream(fields.get("version"))
        ),
        endpoint="/api/v1/text-detection-batches",
        extra_fields=selection if query_selects_model else None,
        field_resolver=(
            None
            if query_selects_model
            else lambda fields: _request_model_selection(
                request,
                fields,
                require_version_for_variant=True,
            )
        ),
    )


@app.post("/api/v1/text-recognitions", tags=["Public pipelines"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/v1/textrecognition", include_in_schema=False)
async def text_recognitions(
    request: Request,
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    query_selects_model = any(name in request.query_params for name in ("version", "model"))
    selection = _model_selection_fields(version, model) if query_selects_model else None
    logger.info(
        "Gateway forwarding recognition endpoint=single version=%s variant=%s upstream=%s",
        (selection or {}).get("version", "<multipart-or-service-default>"),
        (selection or {}).get("model", "<multipart-or-baseline>"),
        REC_SERVICE_URL,
    )
    return await _forward(
        request,
        upstream=REC_SERVICE_URL,
        endpoint="/api/v1/text-recognitions",
        extra_fields=selection,
        field_resolver=(
            None
            if query_selects_model
            else lambda fields: _request_model_selection(request, fields)
        ),
    )


@app.post("/api/v1/text-recognition-batches", tags=["Public pipelines"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def text_recognition_batches(
    request: Request,
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    query_selects_model = any(name in request.query_params for name in ("version", "model"))
    selection = _model_selection_fields(version, model) if query_selects_model else None
    logger.info(
        "Gateway forwarding recognition endpoint=batch version=%s variant=%s upstream=%s",
        (selection or {}).get("version", "<multipart-or-service-default>"),
        (selection or {}).get("model", "<multipart-or-baseline>"),
        REC_SERVICE_URL,
    )
    return await _forward_multiple(
        request,
        upstream=REC_SERVICE_URL,
        endpoint="/api/v1/text-recognition-batches",
        extra_fields=selection,
        field_resolver=(
            None
            if query_selects_model
            else lambda fields: _request_model_selection(request, fields)
        ),
    )


@app.post("/api/v1/table-results", tags=["Public pipelines"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def table_results(request: Request) -> dict[str, Any]:
    return await _forward(
        request,
        upstream=TABLE_PIPELINE_URL,
        endpoint="/api/v1/table-results",
    )


@app.post("/api/v1/table-model-results", tags=["Public pipelines"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def table_model_results(
    request: Request,
    version: str | None = None,
    model: str = "baseline",
    ocr_version: str | None = None,
    profile: str | None = None,
    det_model: str | None = None,
    rec_model: str | None = None,
) -> dict[str, Any]:
    """Expose the notebook-compatible integrated TableRecognitionPipelineV2 output."""
    selection_keys = {"version", "model", "ocr_version", "profile", "det_model", "rec_model"}
    query_selects_model = bool(selection_keys.intersection(request.query_params))
    selection = None
    if query_selects_model:
        selected_version = ocr_version or version
        selected_model = profile or model or "baseline"
        selection = _table_v2_selection_fields(
            selected_version,
            selected_model,
            det_model,
            rec_model,
        )
    return await _forward(
        request,
        upstream=TABLE_MODEL_URL,
        endpoint="/api/v1/table-model-results",
        extra_fields=selection,
        field_resolver=(
            None
            if query_selects_model
            else lambda fields: _request_table_v2_selection(request, fields)
        ),
    )


@app.post("/api/v1/image-verifications", tags=["Public pipelines"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def image_verifications(request: Request) -> dict[str, Any]:
    return await _forward(
        request,
        upstream=IMAGE_VERIFICATION_URL,
        endpoint="/api/v1/image-verifications",
    )


@app.post("/api/v1/image-classifications", tags=["Public model inference"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def image_classifications(request: Request) -> dict[str, Any]:
    """Expose raw SigLIP logits for Backend-owned verification scoring."""
    return await _forward(
        request,
        upstream=SIGLIP_URL,
        endpoint="/api/v1/image-classifications",
    )


@app.get("/api/v1/readiness", tags=["Operations"])
async def readiness(request: Request) -> dict[str, Any]:
    available = _active_upstreams()
    enabled = _pipeline_names("GATEWAY_ENABLED_PIPELINES", default_all=True)
    required = _pipeline_names("GATEWAY_REQUIRED_PIPELINES")
    validate_services(available, enabled, required)

    timeout = _readiness_timeout()
    results = await asyncio.gather(
        *(
            _probe_pipeline(name, available[name], current_request_id=request_id(request), timeout=timeout)
            for name in available
            if name in enabled
        )
    )
    for result in results:
        result["required"] = result["name"] in required
        descriptor = SERVICE_DESCRIPTORS[result["name"]]
        result["service_id"] = descriptor.service_id
        result["kind"] = descriptor.kind

    ready_count = sum(result["status"] == "ready" for result in results)
    failed_required = [result["name"] for result in results if result["required"] and result["status"] != "ready"]
    summary = {
        "enabled": len(results),
        "ready": ready_count,
        "not_ready": len(results) - ready_count,
        "required_not_ready": failed_required,
    }

    if failed_required:
        raise ModelAPIError(
            503,
            "REQUIRED_UPSTREAMS_NOT_READY",
            "One or more required pipeline services are not ready.",
            details=[{"summary": summary, "services": results}],
        )
    if ready_count == 0:
        raise ModelAPIError(
            503,
            "NO_UPSTREAMS_READY",
            "No enabled pipeline service is ready.",
            details=[{"summary": summary, "services": results}],
        )

    status = "ready" if ready_count == len(results) else "degraded"
    return success_response(
        request,
        {"status": status, "summary": summary, "services": results},
        service=SERVICE_NAME,
        model=MODEL_NAME,
    )
