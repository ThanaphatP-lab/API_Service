from __future__ import annotations

import logging
import os

from fastapi import Request

from inference.text_detection import get_model, infer, infer_batch
from core.request_parsing import BATCH_IMAGE_REQUEST_OPENAPI, IMAGE_REQUEST_OPENAPI, parse_image_request
from core.readiness import add_readiness_route
from core.app_factory import create_app
from core.errors import run_image_inference
from shared.contracts import success_response
from shared.model_variants import resolve_model_variant

MODEL_NAME = os.getenv("DET_MODEL_NAME", "PP-OCRv5_server_det")
SERVICE_NAME = "text-detection-model"
app = create_app("Text Detection API", MODEL_NAME, service_name=SERVICE_NAME)
logger = logging.getLogger("uvicorn.error")


@app.post("/api/v1/text-detections", tags=["Model inference"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/v1/textdetection", include_in_schema=False)
@app.post("/predict", include_in_schema=False)
async def predict(
    request: Request,
    version: str | None = None,
    model: str = "baseline",
) -> dict:
    image = await parse_image_request(request)
    try:
        selected = resolve_model_variant(
            "detection",
            version if version is not None else image.fields.get("version"),
            request.query_params.get("model", image.fields.get("model", model)),
        )
        logger.info(
            "Detection request selection endpoint=single version=%s variant=%s model_dir=%s local_weights=%s",
            selected.version,
            selected.variant,
            str(selected.model_dir) if selected.model_dir is not None else "<official-model-cache>",
            selected.model_dir is not None,
        )
        payload = run_image_inference(lambda path: infer(path, selected), image.path)
        response_selection = payload.get("model_selection", selected.public_dict())
        logger.info(
            "Detection response endpoint=single version=%s variant=%s response_model=%s model_name=%s model_dir=%s local_weights=%s region_count=%s",
            selected.version,
            selected.variant,
            selected.response_model,
            response_selection.get("model_name"),
            str(selected.model_dir) if selected.model_dir is not None else "<official-model-cache>",
            selected.model_dir is not None,
            len(payload.get("dt_polys") or []),
        )
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=selected.response_model,
            model_version=selected.version,
        )
    finally:
        image.cleanup()


@app.post("/api/v1/text-detection-batches", tags=["Model inference"], openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def predict_batch(
    request: Request,
    version: str | None = None,
    model: str = "baseline",
) -> dict:
    images = await parse_image_request(request, multiple=True)
    try:
        selected = resolve_model_variant(
            "detection",
            version if version is not None else images.fields.get("version"),
            request.query_params.get("model", images.fields.get("model", model)),
        )
        logger.info(
            "Detection request selection endpoint=batch version=%s variant=%s model_dir=%s local_weights=%s image_count=%s",
            selected.version,
            selected.variant,
            str(selected.model_dir) if selected.model_dir is not None else "<official-model-cache>",
            selected.model_dir is not None,
            len(images.paths),
        )
        payload = run_image_inference(
            lambda _: infer_batch([str(path) for path in images.paths], selected),
            images.paths[0],
        )
        response_selection = payload.get("model_selection", selected.public_dict())
        results = payload.get("results") or []
        region_count = sum(
            len(result.get("dt_polys") or [])
            for result in results
            if isinstance(result, dict)
        )
        logger.info(
            "Detection response endpoint=batch version=%s variant=%s response_model=%s model_name=%s model_dir=%s local_weights=%s image_count=%s result_count=%s region_count=%s",
            selected.version,
            selected.variant,
            selected.response_model,
            response_selection.get("model_name"),
            str(selected.model_dir) if selected.model_dir is not None else "<official-model-cache>",
            selected.model_dir is not None,
            len(images.paths),
            len(results),
            region_count,
        )
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=selected.response_model,
            model_version=selected.version,
        )
    finally:
        images.cleanup()


add_readiness_route(
    app,
    lambda: get_model(resolve_model_variant("detection", None, "baseline")),
)
