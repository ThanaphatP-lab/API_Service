from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shared.contracts import ModelAPIError
from shared.upstream import post_images


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


def detect_text_only(
    image_path: Path,
    *,
    detector_url: str,
    request_id: str,
) -> dict[str, Any]:
    """Expose the original Backend text-detector contract via the pipeline API.

    Unlike ``analyze_document_layout`` this deliberately does not mix layout
    regions (tables/images) into the result. The old table/layout adapters call
    this operation when they need OCR line geometry only.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ModelAPIError(422, "INVALID_IMAGE", "OpenCV could not decode the image.")
    height, width = image.shape[:2]
    detection_data = post_images(
        detector_url,
        "/api/v1/text-detections",
        [image_path],
        request_id=request_id,
    )
    predictions = detection_data.get("predictions") or []
    regions = _text_regions(
        predictions,
        width,
        height,
        expand=False,
        padding=(0, 0, 0, 0),
    )
    legacy_regions = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        legacy_regions.append(
            {
                "text": "",
                "confidence": region["score"],
                "bbox": {
                    "x": x1,
                    "y": y1,
                    "width": round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
                    "x_ratio": region["bbox_ratio"][0],
                    "y_ratio": region["bbox_ratio"][1],
                    "width_ratio": round((x2 - x1) / width, 6),
                    "height_ratio": round((y2 - y1) / height, 6),
                },
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "type": "text",
                "data_type": "text",
                "label": "text",
                "source": "text-detection",
            }
        )
    return {
        "engine": "paddleocr",
        "model": "PP-OCRv5_server_det",
        "image_width": width,
        "image_height": height,
        "regions": legacy_regions,
        "raw_predictions": predictions,
    }


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


def detect_text_batch(
    image_paths: list[Path],
    *,
    detector_url: str,
    request_id: str,
) -> list[dict[str, Any]]:
    if not image_paths:
        return []
    detection_data = post_images(
        detector_url,
        "/api/v1/text-detection-batches",
        image_paths,
        request_id=request_id,
        multiple=True,
    )
    predictions = detection_data.get("predictions") or []
    results: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            results.append({"engine": "paddleocr", "model": "PP-OCRv5_server_det", "image_width": 0, "image_height": 0, "regions": []})
            continue
        height, width = image.shape[:2]
        prediction = predictions[index] if index < len(predictions) and isinstance(predictions[index], dict) else {}
        regions = _text_regions([prediction], width, height, expand=False, padding=(0, 0, 0, 0))
        legacy_regions = []
        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            legacy_regions.append({
                "text": "", "confidence": region["score"],
                "bbox": {"x": x1, "y": y1, "width": round(x2 - x1, 2), "height": round(y2 - y1, 2),
                         "x_ratio": region["bbox_ratio"][0], "y_ratio": region["bbox_ratio"][1],
                         "width_ratio": round((x2 - x1) / width, 6), "height_ratio": round((y2 - y1) / height, 6)},
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "type": "text", "data_type": "text", "label": "text", "source": "text-detection",
            })
        results.append({"engine": "paddleocr", "model": "PP-OCRv5_server_det", "image_width": width, "image_height": height, "regions": legacy_regions, "raw_predictions": [prediction]})
    return results


def analyze_document_layout(
    image_path: Path,
    *,
    layout_url: str,
    detector_url: str,
    request_id: str,
    expand_text_rois: bool,
    auto_roi_mode: str,
    padding: tuple[int, int, int, int],
    table_padding: tuple[int, int, int, int] = (2, 2, 2, 2),
    max_neighbor_overlap: float = 0.15,
) -> dict[str, Any]:
    if auto_roi_mode not in {"text-line", "layout", "hybrid"}:
        raise ModelAPIError(
            422,
            "VALIDATION_ERROR",
            "auto_roi_mode must be text-line, layout, or hybrid.",
            details=[{"field": "auto_roi_mode", "received": auto_roi_mode}],
        )
    image = cv2.imread(str(image_path))
    if image is None:
        raise ModelAPIError(422, "INVALID_IMAGE", "OpenCV could not decode the image.")
    height, width = image.shape[:2]

    layout_data = post_images(
        layout_url,
        "/api/v1/layout-predictions",
        [image_path],
        request_id=request_id,
    )
    detection_data = post_images(
        detector_url,
        "/api/v1/text-detections",
        [image_path],
        request_id=request_id,
    )
    layouts = _layout_regions(layout_data.get("predictions") or [], width, height)
    text_lines = _text_regions(
        detection_data.get("predictions") or [],
        width,
        height,
        expand=False,
        padding=(0, 0, 0, 0),
    )

    structural = [region for region in layouts if _region_type(region) in {"table", "image"}]
    table_boxes = [region["bbox"] for region in structural if _region_type(region) == "table"]
    text_boxes = [region["bbox"] for region in text_lines]
    filtered_texts = [
        text
        for text in text_lines
        if not _belongs_to_table(text["bbox"], table_boxes)
    ]
    filtered_structural = [
        region
        for region in structural
        if _region_type(region) == "table"
        or not _image_contains_text(region["bbox"], text_boxes)
    ]

    prepared_source = [*filtered_texts, *filtered_structural]
    prepared = [
        _prepared_region(
            region,
            width=width,
            height=height,
            expand=expand_text_rois,
            text_padding=padding,
            table_padding=table_padding,
            neighbors=[other["bbox"] for other in prepared_source if other is not region],
            max_neighbor_overlap=max_neighbor_overlap,
        )
        for region in prepared_source
    ]
    prepared_texts = [region for region in prepared if _region_type(region) == "text"]
    prepared_structural = [region for region in prepared if _region_type(region) != "text"]

    if auto_roi_mode == "layout":
        regions = [
            _prepared_region(
                region,
                width=width,
                height=height,
                expand=expand_text_rois,
                text_padding=padding,
                table_padding=table_padding,
                neighbors=[other["bbox"] for other in layouts if other is not region],
                max_neighbor_overlap=max_neighbor_overlap,
            )
            for region in layouts
        ]
    elif auto_roi_mode == "text-line":
        regions = [*prepared_texts, *prepared_structural]
    else:
        regions = [*layouts, *prepared_texts]
    regions.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return {
        "image": {"width": width, "height": height},
        "mode": auto_roi_mode,
        "expand_text_rois": expand_text_rois,
        "regions": regions,
        "counts": {
            "layout": len(layouts),
            "text": len(text_lines),
            "returned": len(regions),
        },
        # The Backend uses these already-known detector boxes for paragraph
        # grouping and can send each crop directly to recognition.
        "text_lines": text_lines,
        "auto_roi_expansion": {
            "enabled": expand_text_rois,
            "padding": {"text": padding, "table": table_padding},
            "max_neighbor_overlap": max_neighbor_overlap,
            "mode": auto_roi_mode,
        },
        "raw": {
            "layout": layout_data.get("predictions") or [],
            "detection": detection_data.get("predictions") or [],
        },
    }
