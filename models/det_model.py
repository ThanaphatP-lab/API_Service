"""Deprecated compatibility imports for text detection inference.

New code must import from :mod:`inference.text_detection`.
"""

from inference.text_detection import (
    get_detector,
    get_model,
    infer,
    infer_batch,
    infer_detection,
    infer_detection_batch,
    predict_detection,
    predict_detection_batch,
    selection_from_settings,
)

__all__ = [
    "get_detector",
    "get_model",
    "infer",
    "infer_batch",
    "infer_detection",
    "infer_detection_batch",
    "predict_detection",
    "predict_detection_batch",
    "selection_from_settings",
]
