from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from clients.model_service_client import ModelClient
from shared.contracts import ModelAPIError
from pipelines.table.grid_analysis import analyze_grid
from pipelines.table.result_assembly import assemble_table_result, assemble_semi_table_result
from pipelines.table.quality import table_result_needs_fallback
from pipelines.table.region_crop import _semi_region_crops


async def recognize_table(
    image_path: Path, *, mode: str, include_ocr: bool,
    wired_url: str, wireless_url: str, ocr_url: str,
    request_id: str, client: ModelClient,
) -> dict[str, Any]:
    """Run grid selection, shared OCR, semi-table recovery, and alternate fallback."""
    if mode not in {"auto", "wired", "wireless"}:
        assemble_table_result(mode=mode, wired=None, wireless=None, ocr=None, grid={"detected": False, "confidence": 0.0})
    grid = await asyncio.to_thread(analyze_grid, image_path)
    initial_model = mode if mode != "auto" else ("wired" if grid.get("detected") else "wireless")
    alternate_model = "wireless" if initial_model == "wired" else "wired"
    service_urls = {"wired": wired_url, "wireless": wireless_url}
    initial_calls: list[tuple[str, object]] = [
        (
            initial_model,
            asyncio.to_thread(
                client.infer,
                service_urls[initial_model],
                "/api/v1/table-structures",
                [image_path],
                request_id=request_id,
            ),
        )
    ]
    if include_ocr:
        initial_calls.append(("ocr", asyncio.to_thread(client.infer, ocr_url, "/api/v1/ocr-results", [image_path], request_id=request_id)))
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
        crop_paths = await asyncio.to_thread(_semi_region_crops, image_path, semi.get("regions") or [])
        try:
            if len(crop_paths) >= 2:
                batch = await asyncio.to_thread(
                    client.infer,
                    service_urls[initial_model],
                    "/api/v1/table-structure-batches",
                    crop_paths,
                    request_id=request_id,
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
                client.infer,
                service_urls[alternate_model],
                "/api/v1/table-structures",
                [image_path],
                request_id=request_id,
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
    return result
