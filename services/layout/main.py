from __future__ import annotations

import os
from typing import Any

from fastapi import Request

from inference.layout_detection import get_model, infer, selection_from_settings
from core.request_parsing import IMAGE_REQUEST_OPENAPI, parse_image_request
from core.readiness import add_readiness_route
from core.app_factory import create_app
from core.errors import run_image_inference
from shared.contracts import success_response

MODEL_NAME = os.getenv("LAYOUT_MODEL_NAME", "PP-DocLayoutV3")
SERVICE_NAME = "layout-model"
app = create_app("PP-DocLayoutV3 API", MODEL_NAME, service_name=SERVICE_NAME)


@app.post("/api/v1/layout-predictions", tags=["Model inference"], openapi_extra=IMAGE_REQUEST_OPENAPI)
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


add_readiness_route(app, lambda: get_model(selection_from_settings()))
