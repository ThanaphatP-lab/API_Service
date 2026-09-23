from __future__ import annotations

import os
from typing import Any

from fastapi import Request
from paddleocr import PaddleOCR

from shared.api import BATCH_IMAGE_REQUEST_OPENAPI, IMAGE_REQUEST_OPENAPI, add_readiness_route, create_app, parse_float, parse_image_request, run_image_inference, singleflight_lru_cache
from shared.contracts import success_response
from shared.serialization import prediction_list
from shared.settings import device, model_dir

DET_MODEL = os.getenv("DET_MODEL_NAME", "PP-OCRv6_medium_det")
REC_MODEL = os.getenv("REC_MODEL_NAME", "th_PP-OCRv5_mobile_rec")
MODEL_NAME = f"{DET_MODEL} + {REC_MODEL}"
SERVICE_NAME = "paddle-ocr-pipeline"
app = create_app("PaddleOCR Thai OCR Pipeline API", MODEL_NAME, service_name=SERVICE_NAME)


@singleflight_lru_cache(maxsize=1)
def model() -> PaddleOCR:
    # This is the requested PaddleOCR pipeline; model directories are optional so
    # a deployment can use either local custom weights or official model names.
    options: dict[str, Any] = {
        "text_detection_model_name": DET_MODEL,
        "text_recognition_model_name": REC_MODEL,
        "use_doc_orientation_classify": False,
        "text_det_unclip_ratio": 2,
        "text_det_thresh": 0.25,
        "text_det_box_thresh": 0.6,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": False,
        "device": device(),
    }
    if directory := model_dir("DET_MODEL_DIR"):
        options["text_detection_model_dir"] = directory
    if directory := model_dir("REC_MODEL_DIR"):
        options["text_recognition_model_dir"] = directory
    return PaddleOCR(**options)


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


def _response_payload(predictions: list[Any], parameters: dict[str, float]) -> dict[str, Any]:
    return {
        "engine": "PaddleOCR",
        "det_model": DET_MODEL,
        "rec_model": REC_MODEL,
        "parameters": parameters,
        "predictions": predictions,
    }


@app.post("/api/v1/ocr-results", tags=["Pipeline"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/predict", include_in_schema=False)
async def predict(request: Request) -> dict[str, Any]:
    image = await parse_image_request(request)
    try:
        parameters = _prediction_parameters(image.fields)
        predictions = run_image_inference(
            lambda p: prediction_list(
                model().predict(
                    p,
                    **parameters,
                )
            ),
            image.path,
        )
        return success_response(
            request,
            _response_payload(predictions, parameters),
            service=SERVICE_NAME,
            model=MODEL_NAME,
        )
    finally:
        image.cleanup()


@app.post("/api/v1/ocr-result-batches", tags=["Pipeline"], openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def predict_batch(request: Request) -> dict[str, Any]:
    """Run the integrated PaddleOCR pipeline for several backend ROI images."""
    images = await parse_image_request(request, multiple=True)
    try:
        parameters = _prediction_parameters(images.fields)
        predictions = run_image_inference(
            lambda _: prediction_list(
                model().predict(
                    input=[str(path) for path in images.paths],
                    **parameters,
                )
            ),
            images.paths[0],
        )
        return success_response(
            request,
            _response_payload(predictions, parameters),
            service=SERVICE_NAME,
            model=MODEL_NAME,
        )
    finally:
        images.cleanup()


add_readiness_route(app, model)
