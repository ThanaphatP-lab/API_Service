from __future__ import annotations

import logging
from typing import Any

from fastapi import Request

from models.table_v2_model import (
    get_table_v2,
    infer_table_v2,
    resolve_table_v2_selection,
)
from shared.api import (
    IMAGE_REQUEST_OPENAPI,
    add_readiness_route,
    create_app,
    parse_image_request,
    run_image_inference,
)
from shared.contracts import success_response


MODEL_NAME = "TableRecognitionPipelineV2"
SERVICE_NAME = "table-recognition-v2-pipeline"
app = create_app("TableRecognitionPipelineV2 API", MODEL_NAME, service_name=SERVICE_NAME)
logger = logging.getLogger("uvicorn.error")


@app.post("/api/v1/table-model-results", tags=["Pipeline"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/predict", include_in_schema=False)
async def predict(
    request: Request,
    version: str | None = None,
    model: str = "baseline",
    ocr_version: str | None = None,
    profile: str | None = None,
    det_model: str | None = None,
    rec_model: str | None = None,
) -> dict[str, Any]:
    image = await parse_image_request(request)
    try:
        selected_version = (
            ocr_version
            or version
            or image.fields.get("ocr_version")
            or image.fields.get("version")
        )
        selected_model = (
            profile
            or request.query_params.get("model")
            or image.fields.get("profile")
            or image.fields.get("model")
            or model
        )
        selected_detection_model = det_model or image.fields.get("det_model")
        selected_recognition_model = rec_model or image.fields.get("rec_model")
        selection = resolve_table_v2_selection(
            selected_version,
            str(selected_model),
            detection_model=(
                str(selected_detection_model)
                if selected_detection_model is not None
                else None
            ),
            recognition_model=(
                str(selected_recognition_model)
                if selected_recognition_model is not None
                else None
            ),
        )
        logger.info(
            "TableV2 request selection version=%s variant=%s det_dir=%s rec_dir=%s",
            selection.version,
            selection.variant,
            (
                str(selection.detection.model_dir)
                if selection.detection.model_dir is not None
                else "<official-model-cache>"
            ),
            (
                str(selection.recognition.model_dir)
                if selection.recognition.model_dir is not None
                else "<official-model-cache>"
            ),
        )
        payload = run_image_inference(
            lambda path: infer_table_v2(
                path,
                selection.version,
                detection_model=selection.detection.variant,
                recognition_model=selection.recognition.variant,
            ),
            image.path,
        )
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=f"{MODEL_NAME}:{selection.variant}",
            model_version=selection.version,
        )
    finally:
        image.cleanup()


add_readiness_route(app, get_table_v2)
