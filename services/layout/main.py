from __future__ import annotations

import os
from typing import Any

from fastapi import Request
from paddleocr import LayoutDetection

from shared.api import IMAGE_REQUEST_OPENAPI, add_readiness_route, create_app, parse_image_request, run_image_inference, singleflight_lru_cache
from shared.contracts import success_response
from shared.inference_adapters import adapt_layout
from shared.settings import device, model_dir

MODEL_NAME = os.getenv("LAYOUT_MODEL_NAME", "PP-DocLayoutV3")
SERVICE_NAME = "layout-model"
app = create_app("PP-DocLayoutV3 API", MODEL_NAME, service_name=SERVICE_NAME)


@singleflight_lru_cache(maxsize=1)
def model() -> LayoutDetection:
    options: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "device": device(),
        "enable_mkldnn": False,
    }
    if directory := model_dir("LAYOUT_MODEL_DIR"):
        options["model_dir"] = directory
    return LayoutDetection(**options)


@app.post("/api/v1/layout-predictions", tags=["Model inference"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/predict", include_in_schema=False)
async def predict(request: Request) -> dict[str, Any]:
    image = await parse_image_request(request)
    try:
        payload = run_image_inference(lambda p: adapt_layout(model().predict(p)), image.path)
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=MODEL_NAME,
        )
    finally:
        image.cleanup()


add_readiness_route(app, model)
