from __future__ import annotations

import asyncio
import logging
from core.settings import runtime_settings
from core.service_settings import ServiceURLs
from typing import Any

from fastapi import Request

from pipelines.ocr.contracts import direct_recognition_contract
from pipelines.ocr.orchestrator import predict_remote_ocr
from clients.model_service_client import HTTPModelClient
from core.request_parsing import BATCH_IMAGE_REQUEST_OPENAPI, IMAGE_REQUEST_OPENAPI, parse_image_request
from core.app_factory import create_app
from shared.contracts import ModelAPIError, request_id, success_response
from shared.model_variants import normalize_model_variant, normalize_model_version

logger = logging.getLogger("uvicorn.error")
MODEL_NAME = "PP-OCRv5_server_det + th_PP-OCRv5_mobile_rec"
SERVICE_NAME = "custom-ocr-pipeline"
service_urls = ServiceURLs()
DET_SERVICE_URL = service_urls.text_detection_url or service_urls.det_service_url
REC_SERVICE_URL = service_urls.rec_service_url
model_client = HTTPModelClient()
app = create_app("Custom Thai OCR Pipeline API", MODEL_NAME, service_name=SERVICE_NAME)


def _recognition_batch_size() -> int:
    return runtime_settings.recognition_batch_size


def _ocr_selection(
    *,
    version: Any,
    model: Any,
    det_model: Any,
    rec_model: Any,
) -> dict[str, str | None]:
    selected_model = normalize_model_variant(model)
    selected_det = normalize_model_variant(det_model if det_model is not None else selected_model)
    selected_rec = normalize_model_variant(rec_model if rec_model is not None else selected_model)
    if version is None or not str(version).strip():
        if any(value != "baseline" for value in (selected_model, selected_det, selected_rec)):
            raise ModelAPIError(
                422,
                "MODEL_VERSION_REQUIRED",
                "version is required when selecting a non-baseline model.",
                details=[{"field": "version"}],
            )
        selected_version = None
    else:
        selected_version = normalize_model_version(version)
    return {
        "version": selected_version,
        "model": selected_model,
        "det_model": selected_det,
        "rec_model": selected_rec,
    }


@app.post("/api/v1/ocr-results", tags=["Pipeline"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/predict", include_in_schema=False)
async def predict(
    request: Request,
    version: str | None = None,
    model: str = "baseline",
    det_model: str | None = None,
    rec_model: str | None = None,
) -> dict:
    image = await parse_image_request(request)
    try:
        selection = _ocr_selection(
            version=version if version is not None else image.fields.get("version"),
            model=request.query_params.get("model", image.fields.get("model", model)),
            det_model=request.query_params.get("det_model", image.fields.get("det_model", det_model)),
            rec_model=request.query_params.get("rec_model", image.fields.get("rec_model", rec_model)),
        )
        logger.info(
            "Custom OCR selection version=%s profile=%s det_variant=%s rec_variant=%s recognition_batch_size=%s",
            selection["version"] or "<service-default>",
            selection["model"],
            selection["det_model"],
            selection["rec_model"],
            _recognition_batch_size(),
        )
        result = await asyncio.to_thread(
            predict_remote_ocr,
            image.path,
            detector_url=DET_SERVICE_URL,
            recognizer_url=REC_SERVICE_URL,
            client=model_client,
            request_id=request_id(request),
            version=selection["version"],
            model=str(selection["model"]),
            det_model=selection["det_model"],
            rec_model=selection["rec_model"],
            recognition_batch_size=_recognition_batch_size(),
        )
        return success_response(
            request,
            result,
            service=SERVICE_NAME,
            model=MODEL_NAME,
        )
    finally:
        image.cleanup()


@app.post("/api/v1/text-recognitions", tags=["Pipeline"], deprecated=True, openapi_extra=IMAGE_REQUEST_OPENAPI)
async def recognize_only(
    request: Request,
    version: str | None = None,
    model: str | None = None,
) -> dict:
    """Recognition-only route used by pre-cropped ROI flows in the old Backend."""
    image = await parse_image_request(request)
    try:
        fields = dict(image.fields)
        if version is not None:
            fields["version"] = version
        if model is not None:
            fields["model"] = model
        logger.info(
            "Deprecated OCR Custom recognition pass-through endpoint=single version=%s variant=%s upstream=%s",
            fields.get("version", "<service-default>"),
            fields.get("model", "baseline"),
            REC_SERVICE_URL,
        )
        data = await asyncio.to_thread(
            model_client.infer,
            REC_SERVICE_URL,
            "/api/v1/text-recognitions",
            [image.path],
            fields=fields,
            request_id=request_id(request),
        )
        predictions = data.get("predictions") if isinstance(data, dict) else []
        result = direct_recognition_contract(predictions[0] if isinstance(predictions, list) and predictions else None)
        selection = data.get("model_selection") if isinstance(data, dict) else None
        if isinstance(selection, dict):
            result["model_selection"] = selection
        response_model = (
            str(selection.get("model_name") or MODEL_NAME)
            if isinstance(selection, dict)
            else MODEL_NAME
        )
        response_version = (
            str(selection["version"])
            if isinstance(selection, dict) and selection.get("version")
            else None
        )
        return success_response(
            request,
            result,
            service=SERVICE_NAME,
            model=response_model,
            model_version=response_version,
        )
    finally:
        image.cleanup()


@app.post("/api/v1/text-recognition-batches", tags=["Pipeline"], deprecated=True, openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def recognize_only_batch(
    request: Request,
    version: str | None = None,
    model: str | None = None,
) -> dict:
    """Batch recognition without running text detection again."""
    images = await parse_image_request(request, multiple=True)
    try:
        fields = dict(images.fields)
        if version is not None:
            fields["version"] = version
        if model is not None:
            fields["model"] = model
        logger.info(
            "Deprecated OCR Custom recognition pass-through endpoint=batch version=%s variant=%s upstream=%s image_count=%s",
            fields.get("version", "<service-default>"),
            fields.get("model", "baseline"),
            REC_SERVICE_URL,
            len(images.paths),
        )
        data = await asyncio.to_thread(
            model_client.infer,
            REC_SERVICE_URL,
            "/api/v1/text-recognition-batches",
            images.paths,
            fields=fields,
            request_id=request_id(request),
            multiple=True,
        )
        predictions = data.get("predictions") if isinstance(data, dict) else []
        normalized = [
            direct_recognition_contract(prediction if isinstance(prediction, dict) else None)
            for prediction in (predictions if isinstance(predictions, list) else [])
        ]
        while len(normalized) < len(images.paths):
            normalized.append(direct_recognition_contract(None))
        selection = data.get("model_selection") if isinstance(data, dict) else None
        result = {"results": normalized, "count": len(normalized)}
        if isinstance(selection, dict):
            result["model_selection"] = selection
        response_model = (
            str(selection.get("model_name") or MODEL_NAME)
            if isinstance(selection, dict)
            else MODEL_NAME
        )
        response_version = (
            str(selection["version"])
            if isinstance(selection, dict) and selection.get("version")
            else None
        )
        return success_response(
            request,
            result,
            service=SERVICE_NAME,
            model=response_model,
            model_version=response_version,
        )
    finally:
        images.cleanup()


@app.get("/api/v1/readiness", tags=["Operations"])
async def readiness(request: Request) -> dict:
    await asyncio.gather(
        asyncio.to_thread(model_client.readiness, DET_SERVICE_URL, request_id=request_id(request)),
        asyncio.to_thread(model_client.readiness, REC_SERVICE_URL, request_id=request_id(request)),
    )
    return success_response(
        request,
        {"status": "ready", "upstreams": [DET_SERVICE_URL, REC_SERVICE_URL]},
        service=SERVICE_NAME,
        model=MODEL_NAME,
    )
