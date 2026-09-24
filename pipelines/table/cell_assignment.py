from __future__ import annotations

from typing import Any
import numpy as np
from pipelines.table.common import (_BORDERLESS_MIN_COLUMNS, _BORDERLESS_MIN_ROWS, _bbox, _clamp01, _intersection_area, _normalize_cells, _normalize_rows, _rows_from_cells, _text)


def _ocr_cells(ocr: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(ocr, dict):
        return []
    result = []
    for segment in ocr.get("segments") or ocr.get("lines") or []:
        if not isinstance(segment, dict) or not _text(segment.get("text")):
            continue
        box = _bbox(segment.get("bbox", segment.get("polygon", segment.get("points"))))
        if not box:
            continue
        try:
            confidence = _clamp01(float(segment.get("confidence", segment.get("rec_score", 0.0)) or 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        result.append(
            {
                "text": _text(segment.get("text")), "confidence": confidence, "bbox": box,
                "center_x": box["x"] + box["width"] / 2,
                "center_y": box["y"] + box["height"] / 2,
            }
        )
    deduplicated: list[dict[str, Any]] = []
    for cell in sorted(result, key=lambda item: (-item["confidence"], item["bbox"]["width"] * item["bbox"]["height"])):
        area = max(1.0, cell["bbox"]["width"] * cell["bbox"]["height"])
        duplicate = False
        for existing in deduplicated:
            existing_area = max(1.0, existing["bbox"]["width"] * existing["bbox"]["height"])
            overlap = _intersection_area(cell["bbox"], existing["bbox"])
            same_text = _text(cell["text"]).casefold() == _text(existing["text"]).casefold()
            if same_text and overlap / min(area, existing_area) >= 0.72:
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(cell)
    return sorted(deduplicated, key=lambda item: (item["center_y"], item["bbox"]["x"]))


def _cluster_ocr(cells: list[dict[str, Any]]) -> tuple[list[list[str]], list[dict[str, Any]]]:
    if not cells:
        return [], []
    row_tolerance = max(8.0, float(np.median([cell["bbox"]["height"] for cell in cells])) * 0.75)
    row_groups: list[list[dict[str, Any]]] = []
    for cell in sorted(cells, key=lambda item: (item["center_y"], item["bbox"]["x"])):
        target = next(
            (row for row in row_groups if abs(cell["center_y"] - sum(item["center_y"] for item in row) / len(row)) <= row_tolerance),
            None,
        )
        if target is None:
            row_groups.append([cell])
        else:
            target.append(cell)
    column_tolerance = max(14.0, float(np.median([cell["bbox"]["width"] for cell in cells])) * 0.55)
    # A global clustering of every center creates fake columns for merged
    # headers. Seed anchors from the densest logical row, then use wide cells
    # as spans over those anchors instead of as new columns.
    densest_group = max(row_groups, key=lambda row: (len(row), -sum(item["center_y"] for item in row) / len(row)))
    anchors: list[float] = []
    for center in sorted(cell["center_x"] for cell in densest_group):
        if not anchors or abs(center - anchors[-1]) > column_tolerance:
            anchors.append(center)
        else:
            anchors[-1] = (anchors[-1] + center) / 2
    if len(row_groups) < _BORDERLESS_MIN_ROWS or len(anchors) < _BORDERLESS_MIN_COLUMNS:
        return [], []
    rows: list[list[str]] = []
    source: list[dict[str, Any]] = []
    for row_index, group in enumerate(row_groups):
        members = [[] for _ in anchors]
        spans: dict[int, int] = {}
        for cell in sorted(group, key=lambda item: item["bbox"]["x"]):
            left = cell["bbox"]["x"]
            right = left + cell["bbox"]["width"]
            covered = [index for index, anchor in enumerate(anchors) if left - column_tolerance * 0.2 <= anchor <= right + column_tolerance * 0.2]
            col = covered[0] if covered else min(range(len(anchors)), key=lambda index: abs(cell["center_x"] - anchors[index]))
            members[col].append(cell)
            spans[col] = max(spans.get(col, 1), len(covered) if covered else 1)
        row = [_text(" ".join(item["text"] for item in values)) for values in members]
        rows.append(row)
        hidden_columns: set[int] = set()
        for col_index, values in enumerate(members):
            if col_index in hidden_columns:
                continue
            span = min(spans.get(col_index, 1), len(anchors) - col_index)
            source.append(
                {
                    "row": row_index, "col": col_index, "text": row[col_index],
                    "ocrText": row[col_index], "groundTruth": row[col_index],
                    "rowSpan": 1, "colSpan": span,
                    "confidence": sum(item["confidence"] for item in values) / len(values) if values else 0.0,
                }
            )
            for hidden_col in range(col_index + 1, col_index + span):
                hidden_columns.add(hidden_col)
                source.append({
                    "row": row_index, "col": hidden_col, "text": "", "ocrText": "", "groundTruth": "",
                    "rowSpan": 1, "colSpan": 1, "hidden": True,
                })
    return _normalize_rows(rows), source


def _grid_from_ocr(cells: list[dict[str, Any]], grid: dict[str, Any]) -> tuple[list[list[str]], list[dict[str, Any]]]:
    horizontal = sorted(float(value) for value in grid.get("horizontal_lines", []))
    vertical = sorted(float(value) for value in grid.get("vertical_lines", []))
    if len(horizontal) < 2 or len(vertical) < 2:
        return [], []
    rows = [["" for _ in range(len(vertical) - 1)] for _ in range(len(horizontal) - 1)]
    assigned: list[list[list[dict[str, Any]]]] = [[[] for _ in row] for row in rows]
    for cell in cells:
        row = next((i for i in range(len(horizontal) - 1) if horizontal[i] <= cell["center_y"] <= horizontal[i + 1]), None)
        col = next((i for i in range(len(vertical) - 1) if vertical[i] <= cell["center_x"] <= vertical[i + 1]), None)
        if row is not None and col is not None:
            assigned[row][col].append(cell)
    source = []
    for row_index, row in enumerate(assigned):
        for col_index, members in enumerate(row):
            members.sort(key=lambda item: item["bbox"]["x"])
            value = _text(" ".join(item["text"] for item in members))
            rows[row_index][col_index] = value
            source.append(
                {
                    "row": row_index, "col": col_index, "text": value,
                    "ocrText": value, "groundTruth": value, "rowSpan": 1, "colSpan": 1,
                    "confidence": sum(item["confidence"] for item in members) / len(members) if members else 0.0,
                }
            )
    return _normalize_rows(rows), source


def _assign_ocr_to_structured_cells(
    structured: dict[str, Any] | None,
    ocr_cells: list[dict[str, Any]],
) -> tuple[list[list[str]], dict[str, Any] | None, dict[str, Any]]:
    """Reapply Thai OCR text to SLANeXt geometry, preserving spans/topology."""
    if not isinstance(structured, dict):
        return [], structured, {"assigned": 0, "unassigned": len(ocr_cells)}
    cells = _normalize_cells(structured.get("cells"))
    owners = [cell for cell in cells if not cell.get("hidden") and isinstance(cell.get("bbox"), dict)]
    if not owners:
        return [], structured, {"assigned": 0, "unassigned": len(ocr_cells)}
    assigned: dict[tuple[int, int], list[dict[str, Any]]] = {
        (cell["row"], cell["col"]): [] for cell in owners
    }
    unassigned = 0
    for ocr_cell in ocr_cells:
        center_x, center_y = ocr_cell["center_x"], ocr_cell["center_y"]
        containing = [
            cell for cell in owners
            if cell["bbox"]["x"] - 2 <= center_x <= cell["bbox"]["x"] + cell["bbox"]["width"] + 2
            and cell["bbox"]["y"] - 2 <= center_y <= cell["bbox"]["y"] + cell["bbox"]["height"] + 2
        ]
        if containing:
            owner = min(containing, key=lambda cell: cell["bbox"]["width"] * cell["bbox"]["height"])
        else:
            overlaps = [(_intersection_area(ocr_cell["bbox"], cell["bbox"]), cell) for cell in owners]
            overlap, owner = max(overlaps, key=lambda item: item[0])
            if overlap <= 0:
                unassigned += 1
                continue
        assigned[(owner["row"], owner["col"])].append(ocr_cell)
    for cell in owners:
        members = sorted(assigned[(cell["row"], cell["col"])], key=lambda item: (item["center_y"], item["bbox"]["x"]))
        if members:
            value = _text(" ".join(member["text"] for member in members))
            cell["text"] = cell["ocrText"] = cell["groundTruth"] = value
            cell["confidence"] = sum(member["confidence"] for member in members) / len(members)
    rows = _rows_from_cells(cells)
    result = {**structured, "rows": rows, "cells": cells}
    assigned_count = sum(bool(value) for value in assigned.values())
    return rows, result, {
        "assigned": assigned_count,
        "unassigned": unassigned,
        "owner_cell_count": len(owners),
        "assignment_ratio": round(assigned_count / max(1, len(ocr_cells)), 4),
    }


def _slice_ocr(ocr: dict[str, Any] | None, region: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(ocr, dict):
        return None
    left, top = float(region["x"]), float(region["y"])
    right, bottom = left + float(region["width"]), top + float(region["height"])
    sliced = []
    for segment in ocr.get("segments") or ocr.get("lines") or []:
        if not isinstance(segment, dict):
            continue
        box = _bbox(segment.get("bbox", segment.get("polygon", segment.get("points"))))
        if not box:
            continue
        center_x = box["x"] + box["width"] / 2.0
        center_y = box["y"] + box["height"] / 2.0
        if not (left <= center_x <= right and top <= center_y <= bottom):
            continue
        shifted = {**segment, "bbox": {**box, "x": box["x"] - left, "y": box["y"] - top}}
        polygon = segment.get("polygon")
        if isinstance(polygon, list):
            shifted["polygon"] = [
                [float(point[0]) - left, float(point[1]) - top]
                for point in polygon
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
        sliced.append(shifted)
    return {**ocr, "segments": sliced, "lines": sliced, "image_width": region["width"], "image_height": region["height"]}

