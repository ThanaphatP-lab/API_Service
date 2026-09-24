from __future__ import annotations

from typing import Any
import numpy as np


def _payload(value: dict[str, Any]) -> dict[str, Any]:
    nested = value.get("res")
    return nested if isinstance(nested, dict) else value


def _bbox_from_points(points: Any) -> list[float] | None:
    array = np.asarray(points, dtype=float)
    if array.shape == (4,):
        x1, y1, x2, y2 = array.tolist()
    elif array.ndim == 2 and array.shape[1] == 2:
        x1, y1 = array.min(axis=0).tolist()
        x2, y2 = array.max(axis=0).tolist()
    else:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def _region(
    bbox: list[float],
    *,
    label: str,
    score: float | None,
    source: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    return {
        "label": label,
        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "bbox_ratio": [
            round(x1 / width, 6),
            round(y1 / height, 6),
            round(x2 / width, 6),
            round(y2 / height, 6),
        ],
        "score": score,
        "source": source,
    }


def _layout_regions(predictions: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for prediction in predictions:
        value = _payload(prediction)
        boxes = value.get("boxes") or value.get("layout_boxes") or []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            bbox = _bbox_from_points(
                box.get("coordinate") or box.get("bbox") or box.get("box") or box.get("points")
            )
            if bbox is None:
                continue
            raw_score = box.get("score", box.get("confidence"))
            score = float(raw_score) if raw_score is not None else None
            regions.append(
                _region(
                    bbox,
                    label=str(box.get("label") or box.get("category") or "layout"),
                    score=score,
                    source="layout",
                    width=width,
                    height=height,
                )
            )
    return regions


def _text_regions(
    predictions: list[dict[str, Any]],
    width: int,
    height: int,
    *,
    expand: bool,
    padding: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    top, right, bottom, left = padding
    for prediction in predictions:
        value = _payload(prediction)
        polygons = value.get("dt_polys") or value.get("polygons") or []
        scores = value.get("dt_scores") or value.get("scores") or []
        for index, polygon in enumerate(polygons):
            bbox = _bbox_from_points(polygon)
            if bbox is None:
                continue
            if expand:
                bbox = [bbox[0] - left, bbox[1] - top, bbox[2] + right, bbox[3] + bottom]
            score = float(scores[index]) if index < len(scores) else None
            regions.append(
                _region(
                    bbox,
                    label="text",
                    score=score,
                    source="text-detection",
                    width=width,
                    height=height,
                )
            )
    return regions


def _contains(outer: list[float], inner: list[float], tolerance: float = 2.0) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _center_inside(inner: list[float], outer: list[float]) -> bool:
    center_x = (inner[0] + inner[2]) / 2.0
    center_y = (inner[1] + inner[3]) / 2.0
    return outer[0] <= center_x <= outer[2] and outer[1] <= center_y <= outer[3]


def _region_type(region: dict[str, Any]) -> str:
    label = str(region.get("label") or "text").lower()
    if "table" in label and "title" not in label and "caption" not in label:
        return "table"
    if any(token in label for token in ("image", "figure", "pic", "seal", "logo", "chart")):
        return "image"
    return "text"


def _belongs_to_table(text_box: list[float], table_boxes: list[list[float]]) -> bool:
    text_area = max(_area(text_box), 1.0)
    return any(
        _center_inside(text_box, table) or _intersection(text_box, table) / text_area >= 0.25
        for table in table_boxes
    )


def _image_contains_text(image_box: list[float], text_boxes: list[list[float]]) -> bool:
    return any(
        _center_inside(text, image_box) or _intersection(text, image_box) / max(_area(text), 1.0) >= 0.35
        for text in text_boxes
    )


def _reduce_neighbor_overlap(
    original: list[float],
    expanded: list[float],
    neighbors: list[list[float]],
    maximum_ratio: float,
) -> list[float]:
    adjusted = expanded[:]
    for neighbor in neighbors:
        denominator = max(1.0, min(_area(adjusted), _area(neighbor)))
        if _intersection(adjusted, neighbor) / denominator <= maximum_ratio:
            continue
        if original[2] <= neighbor[0]:
            adjusted[2] = min(adjusted[2], neighbor[0])
        elif original[0] >= neighbor[2]:
            adjusted[0] = max(adjusted[0], neighbor[2])
        if original[3] <= neighbor[1]:
            adjusted[3] = min(adjusted[3], neighbor[1])
        elif original[1] >= neighbor[3]:
            adjusted[1] = max(adjusted[1], neighbor[3])
        if adjusted[2] <= adjusted[0] or adjusted[3] <= adjusted[1]:
            return original
        denominator = max(1.0, min(_area(adjusted), _area(neighbor)))
        if _intersection(adjusted, neighbor) / denominator > maximum_ratio:
            return original
    return adjusted


def _prepared_region(
    region: dict[str, Any],
    *,
    width: int,
    height: int,
    expand: bool,
    text_padding: tuple[int, int, int, int],
    table_padding: tuple[int, int, int, int],
    neighbors: list[list[float]],
    max_neighbor_overlap: float,
) -> dict[str, Any]:
    original = [float(value) for value in region["bbox"]]
    kind = _region_type(region)
    padding = table_padding if kind == "table" else text_padding
    can_expand = expand and kind in {"text", "table"}
    if can_expand:
        top, right, bottom, left = padding
        expanded = [
            max(0.0, original[0] - left), max(0.0, original[1] - top),
            min(float(width), original[2] + right), min(float(height), original[3] + bottom),
        ]
        final = (
            _reduce_neighbor_overlap(original, expanded, neighbors, max_neighbor_overlap)
            if kind == "text"
            else expanded
        )
    else:
        expanded = original
        final = original
    prepared = _region(
        final,
        label=kind,
        score=region.get("score"),
        source=str(region.get("source") or "layout"),
        width=width,
        height=height,
    )
    prepared["roi_expansion"] = {
        "enabled": can_expand,
        "reason": "table_edge_guard_padding" if can_expand and kind == "table" else "text_padding" if can_expand else "disabled",
        "original_box": original,
        "expanded_box": expanded,
        "final_box": final,
        "padding": {"unit": "px", "top": padding[0], "right": padding[1], "bottom": padding[2], "left": padding[3]},
        "max_neighbor_overlap": max_neighbor_overlap,
        "overlap_adjusted": final != expanded,
    }
    return prepared
