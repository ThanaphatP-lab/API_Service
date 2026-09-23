from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile

import cv2
from fastapi import Request

from pipeline.table_pipeline import analyze_grid, assemble_semi_table_result, assemble_table_result, table_result_needs_fallback
from shared.api import IMAGE_REQUEST_OPENAPI, create_app, parse_bool, parse_image_request
from shared.contracts import ModelAPIError, request_id, success_response
from shared.upstream import get_readiness, post_images


MODEL_NAME = "SLANeXt wired/wireless + OCR"
SERVICE_NAME = "table-recognition-pipeline"
WIRED_SERVICE_URL = os.getenv("TABLE_WIRED_SERVICE_URL", "http://localhost:8007")
WIRELESS_SERVICE_URL = os.getenv("TABLE_WIRELESS_SERVICE_URL", "http://localhost:8008")
OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://localhost:8005")
app = create_app("Table Recognition Pipeline API", MODEL_NAME, service_name=SERVICE_NAME)


def _semi_region_crops(image_path: Path, regions: list[dict]) -> list[Path]:
    source = cv2.imread(str(image_path))
    if source is None:
        return []
    height, width = source.shape[:2]
    paths: list[Path] = []
    for region in regions:
        x = max(0, min(width - 1, int(region.get("x") or 0)))
        y = max(0, min(height - 1, int(region.get("y") or 0)))
        right = min(width, x + max(1, int(region.get("width") or 1)))
        bottom = min(height, y + max(1, int(region.get("height") or 1)))
        crop = source[y:bottom, x:right]
        if crop.size == 0:
            continue
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        handle.close()
        path = Path(handle.name)
        if cv2.imwrite(str(path), crop):
            paths.append(path)
        else:
            path.unlink(missing_ok=True)
    return paths


@app.post("/api/v1/table-results", tags=["Pipeline"], openapi_extra=IMAGE_REQUEST_OPENAPI)
async def table_results(request: Request) -> dict:
    image = await parse_image_request(request)
    try:
        mode = str(image.fields.get("table_mode", "auto")).strip().lower()
        include_ocr = parse_bool(image.fields.get("include_ocr"), field="include_ocr", default=True)
        if mode not in {"auto", "wired", "wireless"}:
            assemble_table_result(mode=mode, wired=None, wireless=None, ocr=None, grid={"detected": False, "confidence": 0.0})
        grid = await asyncio.to_thread(analyze_grid, image.path)
        initial_model = mode if mode != "auto" else ("wired" if grid.get("detected") else "wireless")
        alternate_model = "wireless" if initial_model == "wired" else "wired"
        service_urls = {"wired": WIRED_SERVICE_URL, "wireless": WIRELESS_SERVICE_URL}
        initial_calls: list[tuple[str, object]] = [
            (
                initial_model,
                asyncio.to_thread(
                    post_images,
                    service_urls[initial_model],
                    "/api/v1/table-structures",
                    [image.path],
                    request_id=request_id(request),
                ),
            )
        ]
        if include_ocr:
            initial_calls.append(("ocr", asyncio.to_thread(post_images, OCR_SERVICE_URL, "/api/v1/ocr-results", [image.path], request_id=request_id(request))))
        values = await asyncio.gather(*(call for _, call in initial_calls), return_exceptions=True)
        results: dict[str, dict] = {}
        upstream_failures: list[dict] = []
        for (name, _), value in zip(initial_calls, values):
            if isinstance(value, Exception):
                upstream_failures.append({"upstream": name, "code": value.code if isinstance(value, ModelAPIError) else "UPSTREAM_CALL_FAILED"})
            elif isinstance(value, dict):
                results[name] = value

        fallback_reason = None
        initial_fallback_reason = None
        initial_result = None
        semi_result = None
        structure_inference_calls = 1
        structure_images = 1
        if initial_model in results:
            initial_result = assemble_table_result(
                mode=mode,
                wired=results.get("wired"),
                wireless=results.get("wireless"),
                ocr=results.get("ocr"),
                grid=grid,
            )
            if mode == "auto" and table_result_needs_fallback(initial_result):
                fallback_reason = "initial_candidate_below_legacy_quality_threshold"
                initial_fallback_reason = fallback_reason
        elif mode == "auto":
            fallback_reason = "initial_model_failed"

        semi = grid.get("semi_analysis") if isinstance(grid.get("semi_analysis"), dict) else {}
        if mode == "auto" and fallback_reason and initial_model in results and semi.get("detected"):
            crop_paths = await asyncio.to_thread(_semi_region_crops, image.path, semi.get("regions") or [])
            try:
                if len(crop_paths) >= 2:
                    batch = await asyncio.to_thread(
                        post_images,
                        service_urls[initial_model],
                        "/api/v1/table-structure-batches",
                        crop_paths,
                        request_id=request_id(request),
                        multiple=True,
                    )
                    structure_inference_calls += 1
                    structure_images += len(crop_paths)
                    predictions = batch.get("predictions") if isinstance(batch.get("predictions"), list) else []
                    semi_result = assemble_semi_table_result(
                        mode=mode,
                        model_name=initial_model,
                        predictions=predictions,
                        ocr=results.get("ocr"),
                        grid=grid,
                    )
                    if semi_result is not None and not table_result_needs_fallback(semi_result):
                        fallback_reason = None
            except Exception as value:
                upstream_failures.append({"upstream": f"{initial_model}-semi-batch", "code": value.code if isinstance(value, ModelAPIError) else "UPSTREAM_CALL_FAILED"})
            finally:
                for path in crop_paths:
                    path.unlink(missing_ok=True)

        if mode == "auto" and fallback_reason:
            try:
                results[alternate_model] = await asyncio.to_thread(
                    post_images,
                    service_urls[alternate_model],
                    "/api/v1/table-structures",
                    [image.path],
                    request_id=request_id(request),
                )
                structure_inference_calls += 1
                structure_images += 1
            except Exception as value:
                upstream_failures.append({"upstream": alternate_model, "code": value.code if isinstance(value, ModelAPIError) else "UPSTREAM_CALL_FAILED"})
        if mode == "auto" and alternate_model not in results and semi_result is not None and fallback_reason is None:
            result = semi_result
        elif initial_result is not None and mode != "auto":
            result = initial_result
        else:
            result = assemble_table_result(
                mode=mode,
                wired=results.get("wired"),
                wireless=results.get("wireless"),
                ocr=results.get("ocr"),
                grid=grid,
            )
        if upstream_failures:
            result.setdefault("table_debug", {})["upstream_failures"] = upstream_failures
        result.setdefault("table_debug", {})["auto_strategy"] = {
            "mode": mode,
            "grid_selected_initial_model": initial_model,
            "alternate_model": alternate_model if mode == "auto" else None,
            "alternate_called": alternate_model in results if mode == "auto" else False,
            "fallback_reason": fallback_reason,
            "initial_fallback_reason": initial_fallback_reason,
            "semi_region_batch_called": semi_result is not None,
            "model_inference_count": sum(name in results for name in ("wired", "wireless")),
            "structure_inference_calls": structure_inference_calls,
            "structure_images_processed": structure_images,
            "ocr_called": include_ocr,
        }
        return success_response(request, result, service=SERVICE_NAME, model=MODEL_NAME)
    finally:
        image.cleanup()


@app.get("/api/v1/readiness", tags=["Operations"])
async def readiness(request: Request) -> dict:
    urls = [WIRED_SERVICE_URL, WIRELESS_SERVICE_URL, OCR_SERVICE_URL]
    await asyncio.gather(*(asyncio.to_thread(get_readiness, url, request_id=request_id(request)) for url in urls))
    return success_response(request, {"status": "ready", "upstreams": urls}, service=SERVICE_NAME, model=MODEL_NAME)
