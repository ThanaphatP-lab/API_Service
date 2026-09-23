from __future__ import annotations

import os
from typing import Any

from paddleocr import TextDetection

from shared.api import singleflight_lru_cache
from shared.inference_adapters import adapt_text_detection, adapt_text_detection_batch
from shared.settings import device, model_dir


@singleflight_lru_cache(maxsize=1)
def get_detector() -> TextDetection:
    kwargs: dict[str, Any] = {
        "model_name": os.getenv("DET_MODEL_NAME", "PP-OCRv6_medium_det"),
        "device": device(),
        "enable_mkldnn": False,
    }
    if directory := model_dir("DET_MODEL_DIR"):
        kwargs["model_dir"] = directory
    return TextDetection(**kwargs)


def predict_detection(image_path: str) -> list[dict[str, Any]]:
    """Compatibility interface used by the in-process OCR pipeline."""
    return infer_detection(image_path)["predictions"]


def infer_detection(image_path: str) -> dict[str, Any]:
    """Run only model inference and serialize it to the canonical leaf contract."""
    return adapt_text_detection(get_detector().predict(image_path))


def predict_detection_batch(image_paths: list[str]) -> list[dict[str, Any]]:
    """Compatibility interface preserving one prediction mapping per input image."""
    return infer_detection_batch(image_paths)["predictions"]


def infer_detection_batch(image_paths: list[str]) -> dict[str, Any]:
    if not image_paths:
        return adapt_text_detection_batch([], expected_count=0)
    batch_size = max(1, min(len(image_paths), int(os.getenv("DET_BATCH_SIZE", "1"))))
    output = get_detector().predict(input=image_paths, batch_size=batch_size)
    return adapt_text_detection_batch(output, expected_count=len(image_paths))
