"""Deprecated compatibility imports for text recognition inference.

New code must import from :mod:`inference.text_recognition`.
"""

from inference.text_recognition import (
    get_model,
    get_recognizer,
    infer,
    infer_batch,
    infer_recognition,
    infer_recognition_batch,
    predict_recognition,
    predict_recognition_batch,
    selection_from_settings,
)

__all__ = [
    "get_model",
    "get_recognizer",
    "infer",
    "infer_batch",
    "infer_recognition",
    "infer_recognition_batch",
    "predict_recognition",
    "predict_recognition_batch",
    "selection_from_settings",
]
