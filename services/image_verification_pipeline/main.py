from __future__ import annotations

import asyncio
import os

from fastapi import Request

from pipeline.image_verification_pipeline import normalize_categories, verify_classification_targets
from shared.api import IMAGE_REQUEST_OPENAPI, create_app, parse_image_request, parse_json_field
from shared.contracts import ModelAPIError, request_id, success_response
from shared.upstream import get_readiness, post_images


MODEL_NAME = "SigLIP category verification"
SERVICE_NAME = "image-verification-pipeline"
SIGLIP_SERVICE_URL = os.getenv("SIGLIP_SERVICE_URL", "http://localhost:8009")
app = create_app("Image Verification Pipeline API", MODEL_NAME, service_name=SERVICE_NAME)


@app.post("/api/v1/image-verifications", tags=["Pipeline"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def image_verifications(request: Request) -> dict:
    image = await parse_image_request(request)
    try:
        requested_targets = parse_json_field(image.fields.get("image_categories"), field="image_categories", default=None)
        if requested_targets is None:
            requested_targets = [str(image.fields.get("image_category", "")).strip()]
        if not isinstance(requested_targets, list):
            raise ModelAPIError(422, "VALIDATION_ERROR", "image_categories must be a JSON array.")
        targets = list(dict.fromkeys(str(value).strip() for value in requested_targets if str(value).strip()))
        if not targets:
            raise ModelAPIError(
                422,
                "IMAGE_CATEGORY_REQUIRED",
                "image_category or image_categories is required.",
                details=[{"field": "image_categories", "issue": "required"}],
            )
        categories = normalize_categories(
            parse_json_field(image.fields.get("categories"), field="categories", default=None)
        )
        classification = await asyncio.to_thread(
            post_images,
            SIGLIP_SERVICE_URL,
            "/api/v1/image-classifications",
            [image.path],
            fields={"labels": [item["prompt"] for item in categories]},
            request_id=request_id(request),
        )
        results = verify_classification_targets(
            target_values=targets,
            categories=categories,
            classification=classification,
        )
        selected = next((item for item in results if item.get("passed")), None)
        if selected is None:
            selected = max(results, key=lambda item: float(item.get("evidence_score") or 0.0))
        result = {**selected, "verifications": results, "requested_categories": targets}
        return success_response(request, result, service=SERVICE_NAME, model=MODEL_NAME)
    finally:
        image.cleanup()


@app.get("/api/v1/readiness", tags=["Operations"])
async def readiness(request: Request) -> dict:
    await asyncio.to_thread(get_readiness, SIGLIP_SERVICE_URL, request_id=request_id(request))
    return success_response(
        request,
        {"status": "ready", "upstreams": [SIGLIP_SERVICE_URL]},
        service=SERVICE_NAME,
        model=MODEL_NAME,
    )
