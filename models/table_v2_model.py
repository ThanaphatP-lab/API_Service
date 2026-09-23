from __future__ import annotations

import logging
import os
from typing import Any

from paddleocr import TableRecognitionPipelineV2

from shared.api import singleflight_lru_cache
from shared.inference_adapters import adapt_table
from shared.model_variants import (
    OCRModelPairSpec,
    normalize_model_version,
    resolve_ocr_model_pair,
)
from shared.settings import device, model_dir


logger = logging.getLogger("uvicorn.error")


def _cache_size() -> int:
    try:
        return max(1, int(os.getenv("TABLE_V2_MODEL_CACHE_SIZE", "2")))
    except ValueError:
        return 2


def resolve_table_v2_selection(
    version: str | None = None,
    model: str = "baseline",
    *,
    detection_model: str | None = None,
    recognition_model: str | None = None,
) -> OCRModelPairSpec:
    return resolve_ocr_model_pair(
        version,
        model,
        default_version=os.getenv("TABLE_V2_OCR_VERSION", "v5"),
        detection_variant=detection_model,
        recognition_variant=recognition_model,
    )


@singleflight_lru_cache(maxsize=_cache_size())
def _load_table_v2(
    version: str,
    detection_model: str,
    recognition_model: str,
) -> TableRecognitionPipelineV2:
    """Load the notebook (4) table pipeline once per service process.

    The notebook used CPU. This service intentionally keeps the project's
    MODEL_DEVICE setting, so deployment continues to use gpu:0 by default.
    """
    selected = resolve_table_v2_selection(
        version,
        detection_model=detection_model,
        recognition_model=recognition_model,
    )
    detection = selected.detection
    recognition = selected.recognition
    detection_model_name = detection.model_name
    recognition_model_name = recognition.model_name
    detection_model_dir = (
        str(detection.model_dir) if detection.model_dir is not None else None
    )
    recognition_model_dir = (
        str(recognition.model_dir) if recognition.model_dir is not None else None
    )
    configured_default_version = normalize_model_version(
        os.getenv("TABLE_V2_OCR_VERSION", "v5")
    )

    # Preserve the original TableV2 environment overrides for the baseline.
    # Fine-tuned variants always come from model_variants.json so a generic
    # environment variable cannot silently replace the requested weights.
    if detection.variant == "baseline" and selected.version == configured_default_version:
        detection_model_name = os.getenv(
            "TABLE_V2_TEXT_DETECTION_MODEL_NAME",
            detection_model_name,
        )
        detection_model_dir = (
            model_dir("TABLE_V2_TEXT_DETECTION_MODEL_DIR")
            or detection_model_dir
        )
    if recognition.variant == "baseline" and selected.version == configured_default_version:
        recognition_model_name = os.getenv(
            "TABLE_V2_TEXT_RECOGNITION_MODEL_NAME",
            recognition_model_name,
        )
        recognition_model_dir = (
            model_dir("TABLE_V2_TEXT_RECOGNITION_MODEL_DIR")
            or recognition_model_dir
        )

    kwargs: dict[str, Any] = {
        "wired_table_structure_recognition_model_name": os.getenv(
            "TABLE_V2_WIRED_MODEL_NAME", "SLANeXt_wired"
        ),
        "wireless_table_structure_recognition_model_name": os.getenv(
            "TABLE_V2_WIRELESS_MODEL_NAME", "SLANeXt_wireless"
        ),
        "text_detection_model_name": detection_model_name,
        "text_recognition_model_name": recognition_model_name,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_layout_detection": False,
        "use_ocr_model": True,
        "device": device(),
        "enable_mkldnn": False,
    }
    if detection_model_dir is not None:
        kwargs["text_detection_model_dir"] = detection_model_dir
    if recognition_model_dir is not None:
        kwargs["text_recognition_model_dir"] = recognition_model_dir

    logger.info(
        "Loading TableV2 OCR models version=%s variant=%s det_dir=%s rec_dir=%s",
        selected.version,
        selected.variant,
        detection_model_dir or "<official-model-cache>",
        recognition_model_dir or "<official-model-cache>",
    )

    return TableRecognitionPipelineV2(**kwargs)


def get_table_v2(
    version: str | None = None,
    model: str = "baseline",
    *,
    detection_model: str | None = None,
    recognition_model: str | None = None,
) -> TableRecognitionPipelineV2:
    selected = resolve_table_v2_selection(
        version,
        model,
        detection_model=detection_model,
        recognition_model=recognition_model,
    )
    return _load_table_v2(
        selected.version,
        selected.detection.variant,
        selected.recognition.variant,
    )


def infer_table_v2(
    image_path: str,
    version: str | None = None,
    model: str = "baseline",
    *,
    detection_model: str | None = None,
    recognition_model: str | None = None,
) -> dict[str, Any]:
    """Model inference/serialization only; table quality logic stays upstream."""
    selected = resolve_table_v2_selection(
        version,
        model,
        detection_model=detection_model,
        recognition_model=recognition_model,
    )
    payload = adapt_table(
        get_table_v2(
            selected.version,
            detection_model=selected.detection.variant,
            recognition_model=selected.recognition.variant,
        ).predict(image_path)
    )
    payload["model_selection"] = selected.public_dict()
    return payload
