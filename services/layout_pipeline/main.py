from __future__ import annotations

import asyncio
import os

from fastapi import Request

from pipeline.layout_pipeline import analyze_document_layout, detect_text_batch, detect_text_only
from shared.api import BATCH_IMAGE_REQUEST_OPENAPI, IMAGE_REQUEST_OPENAPI, create_app, parse_bool, parse_image_request
from shared.contracts import request_id, success_response
from shared.upstream import get_readiness


MODEL_NAME = "PP-DocLayoutV3 + text-detection"
SERVICE_NAME = "document-layout-pipeline"
LAYOUT_SERVICE_URL = os.getenv("LAYOUT_SERVICE_URL", "http://localhost:8001")
DET_SERVICE_URL = os.getenv("DET_SERVICE_URL", "http://localhost:8002")
app = create_app("Document Layout Pipeline API", MODEL_NAME, service_name=SERVICE_NAME)


@app.post("/api/v1/text-detections", tags=["Pipeline"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def text_detections(request: Request) -> dict:
    image = await parse_image_request(request)
    try:
        result = await asyncio.to_thread(
            detect_text_only,
            image.path,
            detector_url=DET_SERVICE_URL,
            request_id=request_id(request),
        )
        return success_response(request, result, service=SERVICE_NAME, model=MODEL_NAME)
    finally:
        image.cleanup()


@app.post("/api/v1/text-detection-batches", tags=["Pipeline"], openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def text_detection_batches(request: Request) -> dict:
    images = await parse_image_request(request, multiple=True)
    try:
        results = await asyncio.to_thread(
            detect_text_batch,
            images.paths,
            detector_url=DET_SERVICE_URL,
            request_id=request_id(request),
        )
        return success_response(
            request,
            {"results": results, "count": len(results)},
            service=SERVICE_NAME,
            model=MODEL_NAME,
        )
    finally:
        images.cleanup()


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
        padding = (
            int(os.getenv("AUTO_ROI_EXPAND_TOP_PX", "8")),
            int(os.getenv("AUTO_ROI_EXPAND_RIGHT_PX", "8")),
            int(os.getenv("AUTO_ROI_EXPAND_BOTTOM_PX", "8")),
            int(os.getenv("AUTO_ROI_EXPAND_LEFT_PX", "8")),
        )
        table_padding = (
            int(os.getenv("AUTO_ROI_TABLE_EXPAND_TOP_PX", "2")),
            int(os.getenv("AUTO_ROI_TABLE_EXPAND_RIGHT_PX", "2")),
            int(os.getenv("AUTO_ROI_TABLE_EXPAND_BOTTOM_PX", "2")),
            int(os.getenv("AUTO_ROI_TABLE_EXPAND_LEFT_PX", "2")),
        )
        max_neighbor_overlap = float(os.getenv("AUTO_ROI_MAX_NEIGHBOR_OVERLAP_RATIO", "0.15"))
        result = await asyncio.to_thread(
            analyze_document_layout,
            image.path,
            layout_url=LAYOUT_SERVICE_URL,
            detector_url=DET_SERVICE_URL,
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
        asyncio.to_thread(get_readiness, LAYOUT_SERVICE_URL, request_id=request_id(request)),
        asyncio.to_thread(get_readiness, DET_SERVICE_URL, request_id=request_id(request)),
    )
    return success_response(
        request,
        {"status": "ready", "upstreams": [LAYOUT_SERVICE_URL, DET_SERVICE_URL]},
        service=SERVICE_NAME,
        model=MODEL_NAME,
    )
