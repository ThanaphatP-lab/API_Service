"""Compatibility exports for document layout workflows."""
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
from pipelines.layout.orchestrator import analyze_document_layout, detect_text_only, detect_text_batch
