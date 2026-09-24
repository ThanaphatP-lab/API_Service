from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from paddleocr import PaddleOCR

from core.cache import singleflight_lru_cache
from shared.serialization import prediction_list
from shared.settings import device, model_dir


@dataclass(frozen=True)
class PaddleOCRSelection:
    detection_model_name: str
    recognition_model_name: str
    detection_model_dir: str | None
    recognition_model_dir: str | None
    device: str

    @property
    def model_name(self) -> str:
        return f"{self.detection_model_name} + {self.recognition_model_name}"


def selection_from_settings() -> PaddleOCRSelection:
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


@singleflight_lru_cache(maxsize=1)
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
