from __future__ import annotations

from typing import Any
import numpy as np


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


def _intersection_area(first: dict[str, float], second: dict[str, float]) -> float:
    left = max(first["x"], second["x"])
    top = max(first["y"], second["y"])
    right = min(first["x"] + first["width"], second["x"] + second["width"])
    bottom = min(first["y"] + first["height"], second["y"] + second["height"])
    return max(0.0, right - left) * max(0.0, bottom - top)

