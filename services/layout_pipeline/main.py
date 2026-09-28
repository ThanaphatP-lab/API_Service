from __future__ import annotations

import asyncio
from core.settings import layout_settings
from core.service_settings import ServiceURLs

from fastapi import Request

from pipelines.layout.orchestrator import analyze_document_layout
from clients.model_service_client import HTTPModelClient
from core.request_parsing import IMAGE_REQUEST_OPENAPI, parse_bool, parse_image_request
from core.app_factory import create_app
from shared.contracts import request_id, success_response


MODEL_NAME = "PP-DocLayoutV3 + text-detection"
SERVICE_NAME = "pipeline-document-layout"
service_urls = ServiceURLs()
LAYOUT_SERVICE_URL = service_urls.layout_service_url
DET_SERVICE_URL = service_urls.text_detection_url or service_urls.det_service_url
model_client = HTTPModelClient()
app = create_app("Document Layout Pipeline API", MODEL_NAME, service_name=SERVICE_NAME)


@app.post("/api/v1/document-layouts", tags=["Pipeline"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def document_layouts(request: Request) -> dict:
    image = await parse_image_request(request)
    try:
        expand = parse_bool(
            image.fields.get("expand_text_rois"),
            field="expand_text_rois",
            default=False,
        )
        mode = str(image.fields.get("auto_roi_mode", "text-line")).strip().lower().replace("_", "-")
        padding = layout_settings.padding
        table_padding = layout_settings.table_padding
        max_neighbor_overlap = layout_settings.max_neighbor_overlap
        result = await asyncio.to_thread(
            analyze_document_layout,
            image.path,
            layout_url=LAYOUT_SERVICE_URL,
            detector_url=DET_SERVICE_URL,
            client=model_client,
            request_id=request_id(request),
            expand_text_rois=expand,
            auto_roi_mode=mode,
            padding=padding,
            table_padding=table_padding,
            max_neighbor_overlap=max_neighbor_overlap,
        )
        return success_response(
            request,
            result,
            service=SERVICE_NAME,
            model=MODEL_NAME,
        )
    finally:
        image.cleanup()


@app.get("/api/v1/readiness", tags=["Operations"])
async def readiness(request: Request) -> dict:
    await asyncio.gather(
        asyncio.to_thread(model_client.readiness, LAYOUT_SERVICE_URL, request_id=request_id(request)),
        asyncio.to_thread(model_client.readiness, DET_SERVICE_URL, request_id=request_id(request)),
    )
    return success_response(
        request,
        {"status": "ready", "upstreams": [LAYOUT_SERVICE_URL, DET_SERVICE_URL]},
        service=SERVICE_NAME,
        model=MODEL_NAME,
    )
