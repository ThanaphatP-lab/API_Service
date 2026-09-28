from __future__ import annotations

import asyncio
from core.service_settings import ServiceURLs

from fastapi import Request

from clients.model_service_client import HTTPModelClient
from pipelines.table.orchestrator import recognize_table
from core.request_parsing import IMAGE_REQUEST_OPENAPI, parse_bool, parse_image_request
from core.app_factory import create_app
from shared.contracts import request_id, success_response

MODEL_NAME = "SLANeXt wired/wireless + OCR"
SERVICE_NAME = "pipeline-table-custom"
service_urls = ServiceURLs()
WIRED_SERVICE_URL = service_urls.table_wired_service_url
WIRELESS_SERVICE_URL = service_urls.table_wireless_service_url
OCR_SERVICE_URL = service_urls.ocr_service_url
model_client = HTTPModelClient()
app = create_app("Table Recognition Pipeline API", MODEL_NAME, service_name=SERVICE_NAME)


@app.post("/api/v1/table-results", tags=["Pipeline"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def table_results(request: Request) -> dict:
    image = await parse_image_request(request)
    try:
        mode = str(image.fields.get("table_mode", "auto")).strip().lower()
        include_ocr = parse_bool(image.fields.get("include_ocr"), field="include_ocr", default=True)
        result = await recognize_table(
            image.path, mode=mode, include_ocr=include_ocr,
            wired_url=WIRED_SERVICE_URL, wireless_url=WIRELESS_SERVICE_URL,
            ocr_url=OCR_SERVICE_URL, request_id=request_id(request), client=model_client,
        )
        return success_response(request, result, service=SERVICE_NAME, model=MODEL_NAME)
    finally:
        image.cleanup()


@app.get("/api/v1/readiness", tags=["Operations"])
async def readiness(request: Request) -> dict:
    urls = [WIRED_SERVICE_URL, WIRELESS_SERVICE_URL, OCR_SERVICE_URL]
    await asyncio.gather(*(asyncio.to_thread(model_client.readiness, url, request_id=request_id(request)) for url in urls))
    return success_response(request, {"status": "ready", "upstreams": urls}, service=SERVICE_NAME, model=MODEL_NAME)
