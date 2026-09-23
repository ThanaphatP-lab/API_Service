from __future__ import annotations

import os
from typing import Any

from paddleocr import TextRecognition

from shared.api import singleflight_lru_cache
from shared.inference_adapters import adapt_text_recognition, adapt_text_recognition_batch
from shared.settings import device, model_dir


@singleflight_lru_cache(maxsize=1)
def get_recognizer() -> TextRecognition:
    kwargs: dict[str, Any] = {
        "model_name": os.getenv("REC_MODEL_NAME", "th_PP-OCRv5_mobile_rec"),
        "device": device(),
        "enable_mkldnn": False,
    }
    if directory := model_dir("REC_MODEL_DIR"):
        kwargs["model_dir"] = directory
    return TextRecognition(**kwargs)


def predict_recognition(image_path: str) -> list[dict[str, Any]]:
    """Compatibility interface used by the in-process OCR pipeline."""
    return infer_recognition(image_path)["predictions"]


def infer_recognition(image_path: str) -> dict[str, Any]:
    """Run only model inference and serialize it to the canonical leaf contract."""
    return adapt_text_recognition(get_recognizer().predict(image_path))


def predict_recognition_batch(image_paths: list[str]) -> list[dict[str, Any]]:
    """Compatibility interface preserving one recognition per input image."""
    return infer_recognition_batch(image_paths)["predictions"]


def infer_recognition_batch(image_paths: list[str]) -> dict[str, Any]:
    if not image_paths:
        return adapt_text_recognition_batch([], expected_count=0)
    batch_size = max(1, min(len(image_paths), int(os.getenv("REC_BATCH_SIZE", "8"))))
    output = get_recognizer().predict(input=image_paths, batch_size=batch_size)
    return adapt_text_recognition_batch(output, expected_count=len(image_paths))
