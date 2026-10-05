from __future__ import annotations

import os
from typing import Any

from fastapi import Request

from inference.paddle_ocr import get_model, infer, infer_batch, selection_from_settings
from core.request_parsing import BATCH_IMAGE_REQUEST_OPENAPI, IMAGE_REQUEST_OPENAPI, parse_float, parse_image_request
from core.readiness import add_readiness_route
from core.app_factory import create_app
from core.errors import run_image_inference
from shared.contracts import success_response

DET_MODEL = os.getenv("DET_MODEL_NAME", "PP-OCRv6_medium_det")
REC_MODEL = os.getenv("REC_MODEL_NAME", "th_PP-OCRv5_mobile_rec")
MODEL_NAME = f"{DET_MODEL} + {REC_MODEL}"
SERVICE_NAME = "pipeline-ocr-paddle"
app = create_app("PaddleOCR Thai OCR Pipeline API", MODEL_NAME, service_name=SERVICE_NAME)


def _prediction_parameters(fields: dict[str, Any]) -> dict[str, float]:
    return {
        "text_det_unclip_ratio": parse_float(
            fields.get("text_det_unclip_ratio"),
            field="text_det_unclip_ratio",
            default=2.0,
            minimum=0.1,
            maximum=10.0,
        ),
        "text_det_thresh": parse_float(
            fields.get("text_det_thresh"),
            field="text_det_thresh",
            default=0.25,
            minimum=0.0,
            maximum=1.0,
        ),
        "text_det_box_thresh": parse_float(
            fields.get("text_det_box_thresh"),
            field="text_det_box_thresh",
            default=0.6,
            minimum=0.0,
            maximum=1.0,
        ),
    }


def _selection(request: Request, fields: dict[str, Any]):
    def value(name):
        return request.query_params.get(name, fields.get(name))
    return selection_from_settings(
        value("ocr_version") or value("version"),
        value("profile") or value("model") or "baseline",
        detection_model=value("det_model"), recognition_model=value("rec_model"),
        detection_version=value("det_version"), recognition_version=value("rec_version"),
    )


@app.post("/api/v1/ocr-results", tags=["Pipeline"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/predict", include_in_schema=False)
async def predict(request: Request) -> dict[str, Any]:
    image = await parse_image_request(request)
    try:
        parameters = _prediction_parameters(image.fields)
        selection = _selection(request, image.fields)
        payload = run_image_inference(
            lambda path: infer(path, selection, parameters=parameters),
            image.path,
        )
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=selection.model_name,
        )
    finally:
        image.cleanup()


@app.post("/api/v1/ocr-result-batches", tags=["Pipeline"], openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def predict_batch(request: Request) -> dict[str, Any]:
    """Run the integrated PaddleOCR pipeline for several backend ROI images."""
    images = await parse_image_request(request, multiple=True)
    try:
        parameters = _prediction_parameters(images.fields)
        selection = _selection(request, images.fields)
        payload = run_image_inference(
            lambda _: infer_batch(
                [str(path) for path in images.paths],
                selection,
                parameters=parameters,
            ),
            images.paths[0],
        )
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=selection.model_name,
        )
    finally:
        images.cleanup()


add_readiness_route(app, lambda: get_model(selection_from_settings()))
