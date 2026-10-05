from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any

from paddleocr import PaddleOCR

from core.cache import singleflight_lru_cache
from shared.serialization import prediction_list
from shared.settings import device, model_dir
from shared.model_variants import OCRModelPairSpec, resolve_ocr_model_pair
from core.settings import _positive_int_env


@dataclass(frozen=True)
class PaddleOCRSelection:
    detection_model_name: str
    recognition_model_name: str
    detection_model_dir: str | None
    recognition_model_dir: str | None
    device: str
    pair: OCRModelPairSpec | None = None

    @property
    def model_name(self) -> str:
        return f"{self.detection_model_name} + {self.recognition_model_name}"


def selection_from_settings(version: str | None = None, model: str = "baseline", *,
                            detection_model: str | None = None,
                            recognition_model: str | None = None,
                            detection_version: str | None = None,
                            recognition_version: str | None = None) -> PaddleOCRSelection:
    # Preserve the historical mixed-generation environment selection unless a
    # caller explicitly selects a generation or non-baseline variant.
    if version is not None or model != "baseline" or detection_model or recognition_model or detection_version is not None or recognition_version is not None:
        pair = resolve_ocr_model_pair(
            version, model, default_version=os.getenv("PADDLE_OCR_VERSION", "v5"),
            detection_variant=detection_model, recognition_variant=recognition_model,
            detection_version=detection_version, recognition_version=recognition_version,
        )
        return PaddleOCRSelection(
            pair.detection.model_name, pair.recognition.model_name,
            str(pair.detection.model_dir) if pair.detection.model_dir else None,
            str(pair.recognition.model_dir) if pair.recognition.model_dir else None,
            device(), pair,
        )
    return PaddleOCRSelection(
        detection_model_name=os.getenv("DET_MODEL_NAME", "PP-OCRv6_medium_det"),
        recognition_model_name=os.getenv(
            "REC_MODEL_NAME",
            "th_PP-OCRv5_mobile_rec",
        ),
        detection_model_dir=model_dir("DET_MODEL_DIR"),
        recognition_model_dir=model_dir("REC_MODEL_DIR"),
        device=device(),
    )


@singleflight_lru_cache(maxsize=_positive_int_env("PADDLE_OCR_MODEL_CACHE_SIZE", 2))
def _load_model(selection: PaddleOCRSelection) -> PaddleOCR:
    options: dict[str, Any] = {
        "text_detection_model_name": selection.detection_model_name,
        "text_recognition_model_name": selection.recognition_model_name,
        "use_doc_orientation_classify": False,
        "text_det_unclip_ratio": 2,
        "text_det_thresh": 0.25,
        "text_det_box_thresh": 0.6,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": False,
        "device": selection.device,
    }
    if selection.detection_model_dir is not None:
        options["text_detection_model_dir"] = selection.detection_model_dir
    if selection.recognition_model_dir is not None:
        options["text_recognition_model_dir"] = selection.recognition_model_dir
    logging.getLogger("uvicorn.error").info(
        "paddle_ocr_model_load selection=%s det_name=%s rec_name=%s det_dir=%s rec_dir=%s",
        selection.pair.public_dict() if selection.pair else "legacy-environment",
        selection.detection_model_name, selection.recognition_model_name,
        selection.detection_model_dir or "<official-model-cache>",
        selection.recognition_model_dir or "<official-model-cache>",
    )
    return PaddleOCR(**options)


def get_model(selection: PaddleOCRSelection | None = None) -> PaddleOCR:
    return _load_model(selection or selection_from_settings())


def _payload(
    predictions: list[Any],
    parameters: dict[str, float],
    selection: PaddleOCRSelection,
) -> dict[str, Any]:
    return {
        "engine": "PaddleOCR",
        "det_model": selection.detection_model_name,
        "rec_model": selection.recognition_model_name,
        "parameters": parameters,
        "predictions": predictions,
        "model_selection": selection.pair.public_dict() if selection.pair else {
            "source": "environment",
            "detection": {"model_name": selection.detection_model_name,
                          "local_weights": selection.detection_model_dir is not None},
            "recognition": {"model_name": selection.recognition_model_name,
                            "local_weights": selection.recognition_model_dir is not None},
        },
    }


def infer(
    image_path: str,
    selection: PaddleOCRSelection | None = None,
    *,
    parameters: dict[str, float] | None = None,
) -> dict[str, Any]:
    selected = selection or selection_from_settings()
    selected_parameters = dict(parameters or {})
    predictions = prediction_list(
        get_model(selected).predict(image_path, **selected_parameters)
    )
    return _payload(predictions, selected_parameters, selected)


def infer_batch(
    image_paths: list[str],
    selection: PaddleOCRSelection | None = None,
    *,
    parameters: dict[str, float] | None = None,
) -> dict[str, Any]:
    selected = selection or selection_from_settings()
    selected_parameters = dict(parameters or {})
    predictions = prediction_list(
        get_model(selected).predict(input=image_paths, **selected_parameters)
    )
    return _payload(predictions, selected_parameters, selected)
