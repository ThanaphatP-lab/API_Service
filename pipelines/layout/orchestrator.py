from __future__ import annotations

from pathlib import Path
from typing import Any
import cv2

from shared.contracts import ModelAPIError
from clients.model_service_client import HTTPModelClient, ModelClient
from pipelines.layout.geometry import (
    _payload,
    _bbox_from_points,
    _region,
    _layout_regions,
    _text_regions,
    _contains,
    _area,
    _intersection,
    _center_inside,
    _region_type,
    _belongs_to_table,
    _image_contains_text,
    _reduce_neighbor_overlap,
    _prepared_region,
)


def analyze_document_layout(
    image_path: Path,
    *,
    layout_url: str,
    detector_url: str,
    request_id: str,
    client: ModelClient | None = None,
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

    layout_data = (client if client is not None else HTTPModelClient()).infer(
        layout_url,
        "/api/v1/layout-predictions",
        [image_path],
        request_id=request_id,
    )
    detection_data = (client if client is not None else HTTPModelClient()).infer(
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
