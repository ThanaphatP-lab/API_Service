from __future__ import annotations

from typing import Any
from pipelines.table.common import (_BORDERLESS_MIN_COLUMNS, _TABLE_BORDERLESS_COLUMN_CONSISTENCY_THRESHOLD, _TABLE_BORDERLESS_FILL_RATIO_THRESHOLD, _TABLE_BORDERLESS_FINAL_CONFIDENCE_THRESHOLD, _TABLE_BORDERLESS_SPARSE_ROW_RATIO_THRESHOLD, _TABLE_CANDIDATE_TIE_EPSILON, _TABLE_LOW_OCR_CONFIDENCE_THRESHOLD, _cells_from_rows, _clamp01, _text)
from pipelines.table.html_parser import (_extract_structure)
from pipelines.table.cell_assignment import (_assign_ocr_to_structured_cells, _cluster_ocr, _grid_from_ocr)
from pipelines.table.quality import (_quality)


def _candidate(name: str, data: dict[str, Any], ocr_cells: list[dict[str, Any]], grid: dict[str, Any]) -> dict[str, Any]:
    html, rows, structured = _extract_structure(data)
    original_rows, original_structured = rows, structured
    model_has_text = any(_text(cell) for row in rows for cell in row)
    initial_quality = _quality(rows, structured)
    needs_geometry_recovery = (
        not model_has_text
        or not initial_quality["usable_shape"]
        or initial_quality["column_count"] < _BORDERLESS_MIN_COLUMNS
        or initial_quality["score"] < _TABLE_BORDERLESS_FINAL_CONFIDENCE_THRESHOLD
        or initial_quality["fill_ratio"] < _TABLE_BORDERLESS_FILL_RATIO_THRESHOLD
        or initial_quality["column_consistency"] < _TABLE_BORDERLESS_COLUMN_CONSISTENCY_THRESHOLD
        or initial_quality["sparse_row_ratio"] > _TABLE_BORDERLESS_SPARSE_ROW_RATIO_THRESHOLD
    )
    reconstruction = "model_structure"
    assignment_debug: dict[str, Any] = {"assigned": 0, "unassigned": len(ocr_cells)}
    if ocr_cells and needs_geometry_recovery:
        if model_has_text:
            assigned_rows, assigned_structure = [], structured
        else:
            assigned_rows, assigned_structure, assignment_debug = _assign_ocr_to_structured_cells(structured, ocr_cells)
        grid_rows, grid_cells = _grid_from_ocr(ocr_cells, grid)
        clustered_rows, clustered_cells = _cluster_ocr(ocr_cells)
        if assigned_rows and any(_text(cell) for row in assigned_rows for cell in row):
            rows, structured, source_cells, reconstruction = assigned_rows, assigned_structure, [], "model_structure_ocr"
        elif name == "wired" and grid_rows and any(_text(cell) for row in grid_rows for cell in row):
            rows, source_cells, reconstruction = grid_rows, grid_cells, "coordinate_grid_ocr"
        elif clustered_rows:
            rows, source_cells, reconstruction = clustered_rows, clustered_cells, "borderless_text_clustering"
        else:
            source_cells = []
        if rows and reconstruction != "model_structure_ocr":
            recovered_structure = {"rows": rows, "cells": _cells_from_rows(rows, source_cells), "headerRowCount": 1}
            recovered_quality = _quality(rows, recovered_structure)
            if not model_has_text or recovered_quality["score"] > initial_quality["score"] + _TABLE_CANDIDATE_TIE_EPSILON:
                structured = recovered_structure
            else:
                rows, structured, reconstruction = original_rows, original_structured, "model_structure"
    quality = _quality(rows, structured)
    confidences = [cell["confidence"] for cell in ocr_cells if cell["text"]]
    ocr_score = sum(confidences) / len(confidences) if confidences else 0.0
    final = quality["score"] * (0.65 if confidences else 0.85) + (ocr_score * 0.35 if confidences else 0.0)
    if name == "wired" and grid.get("detected"):
        final += min(0.03, float(grid.get("confidence", 0.0)) * 0.03)
    elif name == "wireless" and not grid.get("detected"):
        final += 0.02
    method = "slanext" if reconstruction in {"model_structure", "model_structure_ocr"} else reconstruction
    final = round(_clamp01(final), 4)
    return {
        "name": name, "method": method, "confidence": final, "rows": rows,
        "structured": structured, "html": html, "raw": data,
        "debug": {
            "quality": quality,
            "ocr_confidence": {
                "available": bool(confidences), "score": round(ocr_score, 4), "average": round(ocr_score, 4),
                "minimum": round(min(confidences), 4) if confidences else 0.0,
                "recognized_count": len(confidences),
                "low_confidence_count": sum(value < _TABLE_LOW_OCR_CONFIDENCE_THRESHOLD for value in confidences),
            },
            "final_confidence": final, "candidate_method": method,
            "model_variant": name, "reconstruction": reconstruction,
            "assignment": assignment_debug,
        },
    }


def _summary(candidate: dict[str, Any]) -> dict[str, Any]:
    quality = candidate["debug"]["quality"]
    return {
        "name": candidate["name"], "method": candidate["method"],
        "structure_score": quality["score"], "ocr_score": candidate["debug"]["ocr_confidence"]["score"],
        "ocr_available": candidate["debug"]["ocr_confidence"]["available"],
        "final_confidence": candidate["confidence"], "quality_score": candidate["confidence"],
        "row_count": quality["row_count"], "column_count": quality["column_count"],
        "usable_shape": quality["usable_shape"], "has_structured_cells": quality["has_structured_cells"],
        "non_empty_cell_count": quality["non_empty_cell_count"], "penalties": quality["penalties"],
    }


def _select(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    valid = [candidate for candidate in candidates if candidate["rows"] or candidate["html"]]
    if not valid:
        return candidates[0], "no_valid_candidate"
    ranked = sorted(valid, key=lambda item: item["confidence"], reverse=True)
    if len(ranked) == 1:
        return ranked[0], "only_valid_candidate"
    if ranked[0]["confidence"] - ranked[1]["confidence"] > _TABLE_CANDIDATE_TIE_EPSILON:
        return ranked[0], "higher_final_confidence"

    def tie_key(candidate: dict[str, Any]) -> tuple[int, int, int, int, int]:
        quality = candidate["debug"]["quality"]
        return (
            int(quality["usable_shape"]), int(quality["has_structured_cells"]),
            int(quality["column_count"] > 1), quality["non_empty_cell_count"],
            int(candidate["method"] == "slanext"),
        )

    selected = max(valid, key=lambda item: (tie_key(item), item["confidence"]))
    return selected, "tie_preferred_structured_slanext" if selected["method"] == "slanext" else "tie_breaker"

