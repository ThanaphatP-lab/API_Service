from __future__ import annotations

import os
from typing import Any

from fastapi import Request
from paddleocr import TableStructureRecognition

from shared.api import BATCH_IMAGE_REQUEST_OPENAPI, IMAGE_REQUEST_OPENAPI, add_readiness_route, create_app, parse_image_request, run_image_inference, singleflight_lru_cache
from shared.contracts import success_response
from shared.inference_adapters import adapt_table_structure
from shared.settings import device, model_dir

MODEL_NAME = os.getenv("TABLE_MODEL_NAME", "SLANeXt_wired")
SERVICE_NAME = "table-structure-model"
app = create_app("Table Structure Recognition API", MODEL_NAME, service_name=SERVICE_NAME)


@singleflight_lru_cache(maxsize=1)
def model() -> TableStructureRecognition:
    options: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "device": device(),
        "enable_mkldnn": False,
    }
    if directory := model_dir("TABLE_MODEL_DIR"):
        options["model_dir"] = directory
    return TableStructureRecognition(**options)


@app.post("/api/v1/table-structures", tags=["Model inference"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/predict", include_in_schema=False)
async def predict(request: Request) -> dict[str, Any]:
    image = await parse_image_request(request)
    try:
        payload = run_image_inference(lambda p: adapt_table_structure(model().predict(p)), image.path)
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
        batch_size = max(1, min(len(images.paths), int(os.getenv("TABLE_BATCH_SIZE", "4"))))
        payload = run_image_inference(
            lambda _: adapt_table_structure(
                model().predict(input=[str(path) for path in images.paths], batch_size=batch_size)
            ),
            images.paths[0],
        )
        payload["count"] = len(images.paths)
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=MODEL_NAME,
        )
    finally:
        images.cleanup()


add_readiness_route(app, model)
