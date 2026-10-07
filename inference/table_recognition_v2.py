from __future__ import annotations

import logging
import os
from typing import Any

from paddleocr import TableRecognitionPipelineV2

from core.cache import singleflight_lru_cache
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


def selection_from_settings(
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
def get_model(selection: OCRModelPairSpec) -> TableRecognitionPipelineV2:
    detection = selection.detection
    recognition = selection.recognition
    detection_model_name = detection.model_name
    recognition_model_name = recognition.model_name
    detection_model_dir = str(detection.model_dir) if detection.model_dir is not None else None
    recognition_model_dir = str(recognition.model_dir) if recognition.model_dir is not None else None
    configured_default_version = normalize_model_version(
        os.getenv("TABLE_V2_OCR_VERSION", "v5")
    )

    # Baseline-only environment overrides remain backward compatible. A
    # fine-tuned selection always comes from the central model catalog.
    if detection.variant == "baseline" and selection.version == configured_default_version:
        detection_model_name = os.getenv(
            "TABLE_V2_TEXT_DETECTION_MODEL_NAME",
            detection_model_name,
        )
        detection_model_dir = model_dir("TABLE_V2_TEXT_DETECTION_MODEL_DIR") or detection_model_dir
    if recognition.variant == "baseline" and selection.version == configured_default_version:
        recognition_model_name = os.getenv(
            "TABLE_V2_TEXT_RECOGNITION_MODEL_NAME",
            recognition_model_name,
        )
        recognition_model_dir = model_dir("TABLE_V2_TEXT_RECOGNITION_MODEL_DIR") or recognition_model_dir

    options: dict[str, Any] = {
        "wired_table_structure_recognition_model_name": os.getenv(
            "TABLE_V2_WIRED_MODEL_NAME",
            "SLANeXt_wired",
        ),
        "wireless_table_structure_recognition_model_name": os.getenv(
            "TABLE_V2_WIRELESS_MODEL_NAME",
            "SLANeXt_wireless",
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
        options["text_detection_model_dir"] = detection_model_dir
    if recognition_model_dir is not None:
        options["text_recognition_model_dir"] = recognition_model_dir

    logger.info(
        "Loading TableV2 OCR models version=%s variant=%s det_dir=%s rec_dir=%s",
        selection.version,
        selection.variant,
        detection_model_dir or "<official-model-cache>",
        recognition_model_dir or "<official-model-cache>",
    )
    for spec, name, directory in (
        (detection, detection_model_name, detection_model_dir),
        (recognition, recognition_model_name, recognition_model_dir),
    ):
        logger.debug(
            "TableV2 effective model load version=%s variant=%s model_name=%s model_dir=%s local_weights=%s",
            spec.version, spec.variant, name,
            directory or "<official-model-cache>", directory is not None,
        )
    return TableRecognitionPipelineV2(**options)


def infer(image_path: str, selection: OCRModelPairSpec) -> dict[str, Any]:
    for component, spec in (("detection", selection.detection), ("recognition", selection.recognition)):
        logger.debug(
            "TableV2 registry selection component=%s version=%s variant=%s model_name=%s model_dir=%s local_weights=%s",
            component, spec.version, spec.variant, spec.model_name,
            str(spec.model_dir) if spec.model_dir is not None else "<official-model-cache>",
            spec.model_dir is not None,
        )
    payload = adapt_table(get_model(selection).predict(image_path))
    payload["model_selection"] = selection.public_dict()
    return payload


# Compatibility API retained while callers migrate from ``models.table_v2_model``.
resolve_table_v2_selection = selection_from_settings


def get_table_v2(
    version: str | None = None,
    model: str = "baseline",
    *,
    detection_model: str | None = None,
    recognition_model: str | None = None,
) -> TableRecognitionPipelineV2:
    return get_model(
        selection_from_settings(
            version,
            model,
            detection_model=detection_model,
            recognition_model=recognition_model,
        )
    )


def infer_table_v2(
    image_path: str,
    version: str | None = None,
    model: str = "baseline",
    *,
    detection_model: str | None = None,
    recognition_model: str | None = None,
) -> dict[str, Any]:
    return infer(
        image_path,
        selection_from_settings(
            version,
            model,
            detection_model=detection_model,
            recognition_model=recognition_model,
        ),
    )
