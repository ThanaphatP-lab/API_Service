from __future__ import annotations

import os
from typing import Any

from fastapi import Request

from inference.table_structure import get_model, infer, infer_batch, selection_from_settings
from core.request_parsing import BATCH_IMAGE_REQUEST_OPENAPI, IMAGE_REQUEST_OPENAPI, parse_image_request
from core.readiness import add_readiness_route
from core.app_factory import create_app
from core.errors import run_image_inference
from shared.contracts import success_response

MODEL_NAME = os.getenv("TABLE_MODEL_NAME", "SLANeXt_wired")
SERVICE_NAME = "table-structure-model"
app = create_app("Table Structure Recognition API", MODEL_NAME, service_name=SERVICE_NAME)


@app.post("/api/v1/table-structures", tags=["Model inference"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/predict", include_in_schema=False)
async def predict(request: Request) -> dict[str, Any]:
    image = await parse_image_request(request)
    try:
        selection = selection_from_settings()
        payload = run_image_inference(lambda path: infer(path, selection), image.path)
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=MODEL_NAME,
        )
    finally:
        image.cleanup()


@app.post("/api/v1/table-structure-batches", tags=["Model inference"], openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def predict_batch(request: Request) -> dict[str, Any]:
    images = await parse_image_request(request, multiple=True)
    try:
        selection = selection_from_settings()
        payload = run_image_inference(
            lambda _: infer_batch([str(path) for path in images.paths], selection),
            images.paths[0],
        )
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=MODEL_NAME,
        )
    finally:
        images.cleanup()


add_readiness_route(app, lambda: get_model(selection_from_settings()))
