"""Compatibility exports for the OCR pipeline."""
from pipelines.ocr.polygon_crop import _crop_quad
from pipelines.ocr.contracts import _value, _text_from_recognition, legacy_ocr_contract, direct_recognition_contract
from pipelines.ocr.orchestrator import predict_custom_ocr, predict_remote_ocr
