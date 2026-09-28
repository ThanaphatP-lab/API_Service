from __future__ import annotations

import os
from typing import Any

from fastapi import Request

from inference.siglip import get_model, infer, selection_from_settings
from core.request_parsing import IMAGE_REQUEST_OPENAPI, parse_image_request
from core.readiness import add_readiness_route
from core.app_factory import create_app
from core.errors import run_image_inference
from shared.contracts import ModelAPIError, success_response
from shared.siglip_categories import siglip_prompts

MODEL_NAME = os.getenv("SIGLIP_MODEL_NAME", "google/siglip-so400m-patch14-384")
SERVICE_NAME = "leaf-image-classification"
app = create_app("SigLIP Zero-shot Classification API", MODEL_NAME, service_name=SERVICE_NAME)


@app.post("/api/v1/image-classifications", tags=["Model inference"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/predict", include_in_schema=False)
async def predict(request: Request) -> dict[str, Any]:
    image = await parse_image_request(request)
    try:
        # Notebook (4) accepts category objects and evaluates only enabled
        # prompts. ``labels`` remains a compatibility alias for the existing
        # image-verification pipeline.
        field_name = "categories" if image.fields.get("categories") is not None else "labels"
        category_value = image.fields.get(field_name)
        try:
            candidates = siglip_prompts(category_value)
        except ValueError as exc:
            raise ModelAPIError(
                422,
                "VALIDATION_ERROR",
                "SigLIP categories are invalid.",
                details=[{"field": field_name, "issue": str(exc)[:200]}],
            ) from exc
        if not candidates:
            raise ModelAPIError(
                422,
                "LABELS_REQUIRED",
                "Provide at least one enabled SigLIP category or label.",
                details=[{"field": field_name, "issue": "required"}],
            )

        selection = selection_from_settings()
        payload = run_image_inference(
            lambda path: infer(path, candidates, selection),
            image.path,
        )
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=MODEL_NAME,
        )
    except Exception as exc:
        if isinstance(exc, ModelAPIError):
            raise
        raise ModelAPIError(
            500,
            "MODEL_INFERENCE_FAILED",
            "SigLIP could not complete inference.",
            details=[{"reason": str(exc)[:500]}],
        ) from exc
    finally:
        image.cleanup()


add_readiness_route(app, lambda: get_model(selection_from_settings()))
