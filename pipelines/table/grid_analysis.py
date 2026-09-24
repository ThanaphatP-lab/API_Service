from __future__ import annotations

from typing import Any
from pathlib import Path
import numpy as np
import cv2
from shared.contracts import ModelAPIError
from pipelines.table.common import (_SEMI_TABLE_MIN_CONFIDENCE, _SEMI_TABLE_MIN_TOPOLOGY_CHANGE_RATIO, _clamp01)


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

