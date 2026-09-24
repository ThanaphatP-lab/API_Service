"""Deprecated compatibility imports for TableRecognitionPipelineV2.

New code must import from :mod:`inference.table_recognition_v2`.
"""

from inference.table_recognition_v2 import (
    get_model,
    get_table_v2,
    infer,
    infer_table_v2,
    resolve_table_v2_selection,
    selection_from_settings,
)

__all__ = [
    "get_model",
    "get_table_v2",
    "infer",
    "infer_table_v2",
    "resolve_table_v2_selection",
    "selection_from_settings",
]
