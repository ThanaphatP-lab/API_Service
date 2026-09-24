from __future__ import annotations

from typing import Any
from pipelines.table.common import (_TABLE_BORDERLESS_COLUMN_CONSISTENCY_THRESHOLD, _TABLE_BORDERLESS_FILL_RATIO_THRESHOLD, _TABLE_BORDERLESS_FINAL_CONFIDENCE_THRESHOLD, _TABLE_BORDERLESS_SPARSE_ROW_RATIO_THRESHOLD, _clamp01, _normalize_rows, _text)


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

