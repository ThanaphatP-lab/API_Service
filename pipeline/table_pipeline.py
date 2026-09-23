from __future__ import annotations

from html import escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shared.contracts import ModelAPIError


# Values from the original table_recognition_v2_adapter.py.
_BORDERLESS_MIN_COLUMNS = 2
_BORDERLESS_MIN_ROWS = 2
_TABLE_BORDERLESS_FINAL_CONFIDENCE_THRESHOLD = 0.72
_TABLE_BORDERLESS_FILL_RATIO_THRESHOLD = 0.20
_TABLE_BORDERLESS_COLUMN_CONSISTENCY_THRESHOLD = 0.45
_TABLE_BORDERLESS_SPARSE_ROW_RATIO_THRESHOLD = 0.70
_TABLE_CANDIDATE_TIE_EPSILON = 0.03
_TABLE_LOW_OCR_CONFIDENCE_THRESHOLD = 0.65
_SEMI_TABLE_MIN_CONFIDENCE = 0.72
_SEMI_TABLE_MIN_TOPOLOGY_CHANGE_RATIO = 0.33


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _normalize_rows(value: Any) -> list[list[str]]:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) for row in value):
        return []
    rows = [[_text(cell) for cell in row] for row in value]
    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def _markdown_table(rows: list[list[str]]) -> str:
    rows = _normalize_rows(rows)
    if not rows:
        return ""

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.replace("|", "/") for cell in row) + " |"

    return "\n".join([fmt(rows[0]), fmt(["---"] * len(rows[0])), *[fmt(row) for row in rows[1:]]])


def _collect_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        result = [value]
        for child in value.values():
            result.extend(_collect_dicts(child))
        return result
    if isinstance(value, (list, tuple)):
        result: list[dict[str, Any]] = []
        for child in value:
            result.extend(_collect_dicts(child))
        return result
    return []


def _find_html(value: Any) -> str:
    if isinstance(value, str):
        return value if "<table" in value.lower() else ""
    if isinstance(value, list) and value and all(isinstance(item, (str, int, float)) for item in value):
        joined = "".join(str(item) for item in value)
        return joined if "<table" in joined.lower() else ""
    if isinstance(value, dict):
        for key in ("table_html", "html", "pred_html", "structure_html", "structure"):
            found = _find_html(value.get(key))
            if found:
                return found
        for child in value.values():
            found = _find_html(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_html(child)
            if found:
                return found
    return ""


class _TableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.cells: list[dict[str, Any]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None
        self.row = -1
        self.col = 0
        self.rowspan = 1
        self.colspan = 1
        self.occupied: set[tuple[int, int]] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self.row += 1
            self.col = 0
            self.current_row = []
        elif tag in {"td", "th"} and self.current_row is not None:
            while (self.row, self.col) in self.occupied:
                self.current_row.append("")
                self.col += 1
            values = {key.lower(): value for key, value in attrs}
            try:
                self.rowspan = max(1, int(values.get("rowspan") or 1))
                self.colspan = max(1, int(values.get("colspan") or 1))
            except (TypeError, ValueError):
                self.rowspan = self.colspan = 1
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self.current_row is not None and self.current_cell is not None:
            value = _text("".join(self.current_cell))
            self.cells.append(
                {
                    "row": self.row, "col": self.col, "text": value,
                    "ocrText": value, "groundTruth": value,
                    "rowSpan": self.rowspan, "colSpan": self.colspan,
                }
            )
            self.current_row.extend([value, *([""] * (self.colspan - 1))])
            for row_offset in range(self.rowspan):
                for col_offset in range(self.colspan):
                    position = (self.row + row_offset, self.col + col_offset)
                    self.occupied.add(position)
                    if row_offset or col_offset:
                        self.cells.append(
                            {
                                "row": position[0], "col": position[1], "text": "",
                                "ocrText": "", "groundTruth": "", "rowSpan": 1,
                                "colSpan": 1, "hidden": True,
                            }
                        )
            self.col += self.colspan
            self.current_cell = None
            self.rowspan = self.colspan = 1
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None


def _structured_from_html(html: str) -> dict[str, Any] | None:
    if not html:
        return None
    parser = _TableHtmlParser()
    try:
        parser.feed(html)
    except Exception:
        return None
    rows = _normalize_rows(parser.rows)
    if not rows:
        return None
    return {"rows": rows, "cells": parser.cells, "headerRowCount": 1, "postProcessing": "stdlib-html-parser"}


def _bbox(value: Any) -> dict[str, float] | None:
    if isinstance(value, dict):
        try:
            x = float(value.get("x", value.get("left", 0.0)))
            y = float(value.get("y", value.get("top", 0.0)))
            if value.get("width") is not None:
                width = float(value["width"])
                height = float(value["height"])
            else:
                width = float(value.get("right", value.get("x2", x))) - x
                height = float(value.get("bottom", value.get("y2", y))) - y
            return {"x": x, "y": y, "width": width, "height": height} if width > 0 and height > 0 else None
        except (TypeError, ValueError, KeyError):
            return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim == 1 and array.size >= 4:
        x1, y1, x2, y2 = array[:4]
    elif array.ndim >= 2 and array.shape[-1] >= 2:
        points = array.reshape(-1, array.shape[-1])
        x1, y1 = np.min(points[:, :2], axis=0)
        x2, y2 = np.max(points[:, :2], axis=0)
    else:
        return None
    width, height = float(x2 - x1), float(y2 - y1)
    return {"x": float(x1), "y": float(y1), "width": width, "height": height} if width > 0 and height > 0 else None


def _normalize_cells(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    cells = []
    for cell in value:
        if not isinstance(cell, dict):
            continue
        row = cell.get("row", cell.get("row_index", cell.get("start_row")))
        col = cell.get("col", cell.get("col_index", cell.get("start_col")))
        if row is None or col is None:
            continue
        try:
            normalized = {
                **cell,
                "row": int(row), "col": int(col),
                "text": _text(cell.get("text", cell.get("content", cell.get("value", "")))),
                "rowSpan": int(cell.get("rowSpan", cell.get("rowspan", cell.get("row_span", 1))) or 1),
                "colSpan": int(cell.get("colSpan", cell.get("colspan", cell.get("col_span", 1))) or 1),
            }
        except (TypeError, ValueError):
            continue
        box = _bbox(cell.get("bbox", cell.get("box", cell.get("coordinate", cell.get("points")))))
        if box:
            normalized["bbox"] = box
        cells.append(normalized)
    return cells


def _rows_from_cells(cells: list[dict[str, Any]]) -> list[list[str]]:
    visible = [cell for cell in cells if not cell.get("hidden")]
    if not visible:
        return []
    max_row = max(cell["row"] + max(1, cell["rowSpan"]) - 1 for cell in visible)
    max_col = max(cell["col"] + max(1, cell["colSpan"]) - 1 for cell in visible)
    rows = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for cell in visible:
        rows[cell["row"]][cell["col"]] = _text(cell.get("text"))
    return rows


def _cells_from_rows(rows: list[list[str]], source: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    by_position = {(cell["row"], cell["col"]): cell for cell in source or []}
    result = []
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            original = by_position.get((row_index, col_index), {})
            text = _text(value)
            result.append(
                {
                    **original, "row": row_index, "col": col_index, "text": text,
                    "ocrText": _text(original.get("ocrText", text)), "groundTruth": text,
                    "rowSpan": int(original.get("rowSpan", 1) or 1),
                    "colSpan": int(original.get("colSpan", 1) or 1),
                }
            )
    return result


def _extract_structure(data: dict[str, Any]) -> tuple[str, list[list[str]], dict[str, Any] | None]:
    html = _find_html(data)
    for record in _collect_dicts(data):
        structured = record.get("table_structured", record.get("structured"))
        if isinstance(structured, dict):
            cells = _normalize_cells(structured.get("cells"))
            rows = _normalize_rows(structured.get("rows")) or _rows_from_cells(cells)
            if rows or cells:
                return html, rows, {**structured, "rows": rows, "cells": cells or _cells_from_rows(rows)}
        cells = _normalize_cells(record.get("cells", record.get("table_cells")))
        rows = _normalize_rows(record.get("table_rows", record.get("rows"))) or _rows_from_cells(cells)
        if rows or cells:
            return html, rows, {"rows": rows, "cells": cells or _cells_from_rows(rows), "headerRowCount": 1}
    parsed = _structured_from_html(html)
    return (html, parsed["rows"], parsed) if parsed else (html, [], None)


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


def _intersection_area(first: dict[str, float], second: dict[str, float]) -> float:
    left = max(first["x"], second["x"])
    top = max(first["y"], second["y"])
    right = min(first["x"] + first["width"], second["x"] + second["width"])
    bottom = min(first["y"] + first["height"], second["y"] + second["height"])
    return max(0.0, right - left) * max(0.0, bottom - top)


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


def _quality(rows: list[list[str]], structured: dict[str, Any] | None) -> dict[str, Any]:
    rows = _normalize_rows(rows)
    row_count = len(rows)
    column_count = max((len(row) for row in rows), default=0)
    counts = [sum(bool(_text(cell)) for cell in row) for row in rows]
    total = row_count * column_count
    non_empty = sum(counts)
    fill_ratio = non_empty / total if total else 0.0
    non_empty_row_ratio = sum(count > 0 for count in counts) / row_count if row_count else 0.0
    active = [count for count in counts if count]
    if column_count and active:
        average = sum(active) / len(active)
        variance = sum((count - average) ** 2 for count in active) / len(active)
        column_consistency = _clamp01(1.0 - variance ** 0.5 / column_count)
    else:
        column_consistency = 0.0
    sparse_ratio = sum(0 < count / max(column_count, 1) < 0.35 for count in counts) / row_count if row_count else 0.0
    cells = structured.get("cells", []) if isinstance(structured, dict) else []
    visible_cells = [cell for cell in cells if isinstance(cell, dict) and not cell.get("hidden")]
    has_structured = bool(visible_cells)
    merged_cells = [
        cell for cell in visible_cells
        if int(cell.get("rowSpan", cell.get("rowspan", 1)) or 1) > 1
        or int(cell.get("colSpan", cell.get("colspan", 1)) or 1) > 1
    ]
    merged_ratio = len(merged_cells) / len(visible_cells) if visible_cells else 0.0
    merged_adjustment = 0.04 if 0.0 < merged_ratio <= 0.35 else (-0.04 if merged_ratio > 0.65 else 0.0)
    usable = row_count >= 2 and column_count >= 2 and sum(count > 0 for count in counts) >= 2
    shape_score = 1.0 if usable else (0.35 if row_count and column_count else 0.0)
    score = shape_score * 0.30 + fill_ratio * 0.24 + non_empty_row_ratio * 0.16 + column_consistency * 0.18 + (1 - sparse_ratio) * 0.08 + (0.08 if has_structured else 0.0) + merged_adjustment
    if not usable:
        score *= 0.65
    penalties = [
        name for name, condition in (
            ("no_rows", not rows), ("unusable_shape", not usable), ("too_few_columns", column_count < 2),
            ("low_fill_ratio", fill_ratio < _TABLE_BORDERLESS_FILL_RATIO_THRESHOLD),
            ("low_column_consistency", column_consistency < _TABLE_BORDERLESS_COLUMN_CONSISTENCY_THRESHOLD),
            ("too_many_sparse_rows", sparse_ratio > _TABLE_BORDERLESS_SPARSE_ROW_RATIO_THRESHOLD),
        ) if condition
    ]
    return {
        "score": round(_clamp01(score), 4), "row_count": row_count, "column_count": column_count,
        "non_empty_cell_count": non_empty, "fill_ratio": round(fill_ratio, 4),
        "non_empty_row_ratio": round(non_empty_row_ratio, 4),
        "column_consistency": round(column_consistency, 4), "sparse_row_ratio": round(sparse_ratio, 4),
        "has_structured_cells": has_structured, "merged_cell_ratio": round(merged_ratio, 4),
        "usable_shape": usable, "penalties": penalties,
    }


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


def _line_groups(projection: np.ndarray, threshold: float) -> list[int]:
    positions = np.where(projection >= threshold)[0].tolist()
    groups: list[list[int]] = []
    for position in positions:
        if not groups or position - groups[-1][-1] > 3:
            groups.append([position])
        else:
            groups[-1].append(position)
    return [int(sum(group) / len(group)) for group in groups]


def _semi_table_regions(vertical_mask: np.ndarray, horizontal_lines: list[int], *, width: int, height: int) -> dict[str, Any]:
    boundaries = sorted(set([0, *horizontal_lines, height]))
    minimum_band_height = max(18, int(height * 0.06))
    bands = [(top, bottom) for top, bottom in zip(boundaries, boundaries[1:]) if bottom - top >= minimum_band_height]
    if len(bands) < 2:
        return {"detected": False, "confidence": 0.0, "regions": [], "reason": "insufficient_bands"}
    counts = []
    for top, bottom in bands:
        positions = _line_groups(np.mean(vertical_mask[top:bottom] > 0, axis=0), 0.18)
        counts.append(len([position for position in positions if 2 <= position <= width - 3]))
    changes = sum(left != right for left, right in zip(counts, counts[1:]))
    distinct = len(set(counts))
    change_ratio = changes / max(1, len(bands) - 1)
    if not changes or distinct < 2:
        return {"detected": False, "confidence": 0.0, "regions": [], "reason": "single_topology", "segment_counts": counts, "topology_change_ratio": round(change_ratio, 4)}
    grouped: list[dict[str, Any]] = []
    current_top, current_bottom = bands[0]
    current_count = counts[0]
    for (top, bottom), count in zip(bands[1:], counts[1:]):
        if count != current_count:
            grouped.append({"x": 0, "y": current_top, "width": width, "height": current_bottom - current_top, "grid_line_count": current_count})
            current_top, current_count = top, count
        current_bottom = bottom
    grouped.append({"x": 0, "y": current_top, "width": width, "height": current_bottom - current_top, "grid_line_count": current_count})
    regions = [region for region in grouped if region["height"] >= max(24, int(height * 0.12))]
    confidence = _clamp01(0.45 + min(0.3, change_ratio * 0.45) + min(0.25, len(horizontal_lines) / 12.0))
    detected = len(regions) >= 2 and confidence >= _SEMI_TABLE_MIN_CONFIDENCE and change_ratio >= _SEMI_TABLE_MIN_TOPOLOGY_CHANGE_RATIO
    return {
        "detected": detected, "confidence": round(confidence, 4), "regions": regions if detected else [],
        "reason": "topology_change_detected" if detected else "low_topology_confidence",
        "segment_counts": counts, "topology_change_ratio": round(change_ratio, 4), "distinct_topologies": distinct,
    }


def analyze_grid(image_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ModelAPIError(422, "INVALID_IMAGE", "OpenCV could not decode the table image.")
    height, width = image.shape
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, width // 24), 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, height // 24))))
    horizontal_lines = _line_groups(np.mean(horizontal > 0, axis=1), 0.25)
    vertical_lines = _line_groups(np.mean(vertical > 0, axis=0), 0.25)
    intersection_ratio = float(np.count_nonzero(cv2.bitwise_and(horizontal, vertical))) / max(1, height * width)
    confidence = min(1.0, len(horizontal_lines) / 4 * 0.4 + len(vertical_lines) / 3 * 0.4 + min(intersection_ratio * 500, 1.0) * 0.2)
    semi_analysis = _semi_table_regions(vertical, horizontal_lines, width=width, height=height)
    return {
        "detected": len(horizontal_lines) >= 2 and len(vertical_lines) >= 2,
        "confidence": round(confidence, 6), "horizontal_lines": horizontal_lines,
        "vertical_lines": vertical_lines, "intersection_ratio": round(intersection_ratio, 8),
        "line_summary": {"horizontal": len(horizontal_lines), "vertical": len(vertical_lines)},
        "image_width": width, "image_height": height, "semi_analysis": semi_analysis,
    }


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


def table_result_needs_fallback(result: dict[str, Any], minimum_confidence: float = _TABLE_BORDERLESS_FINAL_CONFIDENCE_THRESHOLD) -> bool:
    """Decide whether auto mode should pay for the alternate structure model."""
    rows = result.get("table_rows") if isinstance(result.get("table_rows"), list) else []
    structured = result.get("table_structured") if isinstance(result.get("table_structured"), dict) else {}
    candidates = result.get("table_candidates") if isinstance(result.get("table_candidates"), list) else []
    selected = candidates[0] if len(candidates) == 1 and isinstance(candidates[0], dict) else None
    if not rows and not structured.get("cells"):
        return True
    if float(result.get("quality_score") or result.get("confidence") or 0.0) < minimum_confidence:
        return True
    if selected is not None:
        if not selected.get("usable_shape"):
            return True
        if int(selected.get("column_count") or 0) < 2:
            return True
        if selected.get("penalties"):
            return True
    return False


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


def _slice_grid(grid: dict[str, Any], region: dict[str, Any]) -> dict[str, Any]:
    left, top = float(region["x"]), float(region["y"])
    right, bottom = left + float(region["width"]), top + float(region["height"])
    horizontal = [float(value) - top for value in grid.get("horizontal_lines", []) if top <= float(value) <= bottom]
    vertical = [float(value) - left for value in grid.get("vertical_lines", []) if left <= float(value) <= right]
    return {
        **grid,
        "horizontal_lines": horizontal,
        "vertical_lines": vertical,
        "detected": len(horizontal) >= 2 and len(vertical) >= 2,
        "line_summary": {"horizontal": len(horizontal), "vertical": len(vertical)},
        "image_width": region["width"], "image_height": region["height"],
    }


def _rows_html(rows: list[list[str]]) -> str:
    return "<table>" + "".join(
        "<tr>" + "".join(f"<td>{escape(_text(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    ) + "</table>"


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
