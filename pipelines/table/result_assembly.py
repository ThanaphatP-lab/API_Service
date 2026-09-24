from __future__ import annotations

from typing import Any
from shared.contracts import ModelAPIError
from pipelines.table.common import (_SEMI_TABLE_MIN_CONFIDENCE, _SEMI_TABLE_MIN_TOPOLOGY_CHANGE_RATIO, _TABLE_BORDERLESS_COLUMN_CONSISTENCY_THRESHOLD, _TABLE_BORDERLESS_FILL_RATIO_THRESHOLD, _TABLE_BORDERLESS_FINAL_CONFIDENCE_THRESHOLD, _TABLE_BORDERLESS_SPARSE_ROW_RATIO_THRESHOLD, _TABLE_CANDIDATE_TIE_EPSILON, _TABLE_LOW_OCR_CONFIDENCE_THRESHOLD, _cells_from_rows, _markdown_table, _normalize_cells, _normalize_rows)
from pipelines.table.html_parser import (_rows_html)
from pipelines.table.cell_assignment import (_ocr_cells, _slice_ocr)
from pipelines.table.quality import (_quality)
from pipelines.table.candidates import (_candidate, _select, _summary)
from pipelines.table.grid_analysis import (_slice_grid)


def assemble_table_result(
    *,
    mode: str,
    wired: dict[str, Any] | None,
    wireless: dict[str, Any] | None,
    ocr: dict[str, Any] | None,
    grid: dict[str, Any],
) -> dict[str, Any]:
    if mode not in {"auto", "wired", "wireless"}:
        raise ModelAPIError(422, "VALIDATION_ERROR", "table_mode must be auto, wired, or wireless.", details=[{"field": "table_mode", "received": mode}])
    geometry = _ocr_cells(ocr)
    candidates = []
    if wired is not None:
        candidates.append(_candidate("wired", wired, geometry, grid))
    if wireless is not None:
        candidates.append(_candidate("wireless", wireless, geometry, grid))
    if not candidates:
        raise ModelAPIError(502, "TABLE_CANDIDATES_EMPTY", "No table model returned a candidate.")
    selected, reason = _select(candidates)
    summaries = [_summary(candidate) for candidate in candidates]
    rows, confidence = selected["rows"], selected["confidence"]
    ocr_text = str(ocr.get("text") or "") if isinstance(ocr, dict) else ""
    segments = ocr.get("segments", []) if isinstance(ocr, dict) else []
    table_debug = {
        **selected["debug"], "status": "success" if rows else "empty", "grid_analysis": grid,
        "candidate_competition": {
            "selected_method": selected["method"], "selected_model": selected["name"],
            "selection_reason": reason, "candidate_count": len(candidates), "candidates": summaries,
        },
        "legacy_thresholds": {
            "borderless_final_confidence": _TABLE_BORDERLESS_FINAL_CONFIDENCE_THRESHOLD,
            "borderless_fill_ratio": _TABLE_BORDERLESS_FILL_RATIO_THRESHOLD,
            "borderless_column_consistency": _TABLE_BORDERLESS_COLUMN_CONSISTENCY_THRESHOLD,
            "borderless_sparse_row_ratio": _TABLE_BORDERLESS_SPARSE_ROW_RATIO_THRESHOLD,
            "candidate_tie_epsilon": _TABLE_CANDIDATE_TIE_EPSILON,
            "low_ocr_confidence": _TABLE_LOW_OCR_CONFIDENCE_THRESHOLD,
            "semi_min_confidence": _SEMI_TABLE_MIN_CONFIDENCE,
            "semi_min_topology_change_ratio": _SEMI_TABLE_MIN_TOPOLOGY_CHANGE_RATIO,
        },
    }
    legacy = {
        "text": _markdown_table(rows) if rows else ocr_text, "confidence": confidence,
        "segments": segments,
        "attempts": [{"step": "slanext_wired_wireless_competition", "selection_reason": reason}],
        "preprocessing": "table_recognition_v2_microservice_pipeline",
        "engine": "table_recognition_v2",
        "model": f"SLANeXt_{selected['name']}/th_PP-OCRv5_mobile_rec",
        "table_html": selected["html"], "table_rows": rows,
        "table_structured": selected["structured"], "table_debug": table_debug,
        "table_selected_method": selected["method"], "table_candidates": summaries,
        "table_semi_analysis": grid.get("semi_analysis") or {
            "detected": False, "confidence": 0.0, "regions": [],
            "line_summary": grid.get("line_summary", {}),
        },
    }
    return {
        **legacy, "table_mode": mode, "selected_method": selected["method"],
        "selected_model": selected["name"], "quality_score": confidence,
        "structure": selected["raw"], "ocr": ocr, "grid_analysis": grid,
        "candidates": summaries,
    }


def assemble_semi_table_result(
    *,
    mode: str,
    model_name: str,
    predictions: list[dict[str, Any]],
    ocr: dict[str, Any] | None,
    grid: dict[str, Any],
) -> dict[str, Any] | None:
    semi = grid.get("semi_analysis") if isinstance(grid.get("semi_analysis"), dict) else {}
    regions = semi.get("regions") if isinstance(semi.get("regions"), list) else []
    section_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, region in enumerate(regions):
        if index >= len(predictions) or not isinstance(region, dict) or not isinstance(predictions[index], dict):
            continue
        model_payload = {"predictions": [predictions[index]]}
        result = assemble_table_result(
            mode=mode,
            wired=model_payload if model_name == "wired" else None,
            wireless=model_payload if model_name == "wireless" else None,
            ocr=_slice_ocr(ocr, region),
            grid=_slice_grid(grid, region),
        )
        if result.get("table_rows"):
            section_results.append((region, result))
    if len(section_results) < 2:
        return None
    merged_rows: list[list[str]] = []
    merged_cells: list[dict[str, Any]] = []
    sections = []
    row_offset = 0
    confidences = []
    for index, (region, result) in enumerate(section_results):
        rows = _normalize_rows(result.get("table_rows"))
        structured = result.get("table_structured") if isinstance(result.get("table_structured"), dict) else {}
        for cell in _normalize_cells(structured.get("cells")):
            shifted = {**cell, "row": cell["row"] + row_offset}
            if isinstance(shifted.get("bbox"), dict):
                shifted["bbox"] = {
                    **shifted["bbox"],
                    "x": float(shifted["bbox"]["x"]) + float(region["x"]),
                    "y": float(shifted["bbox"]["y"]) + float(region["y"]),
                }
            merged_cells.append(shifted)
        merged_rows.extend(rows)
        confidences.append(float(result.get("confidence") or 0.0))
        sections.append({"index": index, "bbox": region, "row_start": row_offset, "row_count": len(rows), "confidence": result.get("confidence"), "selected_method": result.get("selected_method")})
        row_offset += len(rows)
    merged_rows = _normalize_rows(merged_rows)
    structured = {"rows": merged_rows, "cells": merged_cells or _cells_from_rows(merged_rows), "headerRowCount": 1, "sections": sections}
    quality = _quality(merged_rows, structured)
    confidence = round(sum(confidences) / len(confidences), 4) if confidences else quality["score"]
    return {
        "text": _markdown_table(merged_rows), "confidence": confidence,
        "segments": ocr.get("segments", []) if isinstance(ocr, dict) else [],
        "attempts": [{"step": "semi_table_region_batch", "region_count": len(section_results)}],
        "preprocessing": "table_recognition_v2_semi_region_batch",
        "engine": "table_recognition_v2", "model": f"SLANeXt_{model_name}/th_PP-OCRv5_mobile_rec",
        "table_html": _rows_html(merged_rows), "table_rows": merged_rows, "table_structured": structured,
        "table_debug": {"semi_region_batch": True, "sections": sections, "quality": quality},
        "table_selected_method": "semi_table_region_batch", "table_candidates": [],
        "table_semi_analysis": semi, "table_mode": mode, "selected_method": "semi_table_region_batch",
        "selected_model": model_name, "quality_score": quality["score"], "structure": {"predictions": predictions},
        "ocr": ocr, "grid_analysis": grid, "candidates": [], "table_sections": sections,
    }

