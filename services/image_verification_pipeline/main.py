from __future__ import annotations

import asyncio
from core.service_settings import ServiceURLs

from fastapi import Request

from pipelines.verification.scoring import normalize_categories
from pipelines.verification.orchestrator import verify_image
from clients.model_service_client import HTTPModelClient
from core.request_parsing import IMAGE_REQUEST_OPENAPI, parse_image_request, parse_json_field
from core.app_factory import create_app
from shared.contracts import ModelAPIError, request_id, success_response


MODEL_NAME = "SigLIP category verification"
SERVICE_NAME = "image-verification-pipeline"
service_urls = ServiceURLs()
SIGLIP_SERVICE_URL = service_urls.siglip_service_url
model_client = HTTPModelClient()
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
        result = await asyncio.to_thread(
            verify_image,
            image.path,
            targets=targets,
            categories=categories,
            classifier_url=SIGLIP_SERVICE_URL,
            client=model_client,
            request_id=request_id(request),
        )
        return success_response(request, result, service=SERVICE_NAME, model=MODEL_NAME)
    finally:
        image.cleanup()


@app.get("/api/v1/readiness", tags=["Operations"])
async def readiness(request: Request) -> dict:
    await asyncio.to_thread(model_client.readiness, SIGLIP_SERVICE_URL, request_id=request_id(request))
    return success_response(
        request,
        {"status": "ready", "upstreams": [SIGLIP_SERVICE_URL]},
        service=SERVICE_NAME,
        model=MODEL_NAME,
    )
