from __future__ import annotations

import logging
import os
from typing import Any

from paddleocr import TextRecognition

from shared.api import singleflight_lru_cache
from shared.inference_adapters import adapt_text_recognition, adapt_text_recognition_batch
from shared.model_variants import resolve_model_variant
from shared.settings import device


# Use Uvicorn's configured logger so INFO messages are visible both in the
# foreground console and in model-stack.sh redirected service logs.
logger = logging.getLogger("uvicorn.error")


@singleflight_lru_cache(maxsize=8)
def get_recognizer(version: str | None = None, model: str = "baseline") -> TextRecognition:
    selected = resolve_model_variant("recognition", version, model)
    selected_device = device()
    kwargs: dict[str, Any] = {
        "model_name": selected.model_name,
        "device": selected_device,
        "enable_mkldnn": False,
    }
    if selected.model_dir is not None:
        kwargs["model_dir"] = str(selected.model_dir)

    logger.info(
        "Loading recognition model version=%s variant=%s model_name=%s model_dir=%s device=%s",
        selected.version,
        selected.variant,
        selected.model_name,
        str(selected.model_dir) if selected.model_dir is not None else "<official-model-cache>",
        selected_device,
    )
    recognizer = TextRecognition(**kwargs)
    logger.info(
        "Recognition model loaded version=%s variant=%s model_name=%s local_weights=%s",
        selected.version,
        selected.variant,
        selected.model_name,
        selected.model_dir is not None,
    )
    return recognizer


def predict_recognition(
    image_path: str,
    version: str | None = None,
    model: str = "baseline",
) -> list[dict[str, Any]]:
    """Compatibility interface used by the in-process OCR pipeline."""
    return infer_recognition(image_path, version, model)["predictions"]


def infer_recognition(
    image_path: str,
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    """Run only model inference and serialize it to the canonical leaf contract."""
    selected = resolve_model_variant("recognition", version, model)
    payload = adapt_text_recognition(
        get_recognizer(selected.version, selected.variant).predict(image_path)
    )
    payload["model_selection"] = selected.public_dict()
    return payload


def predict_recognition_batch(
    image_paths: list[str],
    version: str | None = None,
    model: str = "baseline",
) -> list[dict[str, Any]]:
    """Compatibility interface preserving one recognition per input image."""
    return infer_recognition_batch(image_paths, version, model)["predictions"]


def infer_recognition_batch(
    image_paths: list[str],
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    selected = resolve_model_variant("recognition", version, model)
    if not image_paths:
        payload = adapt_text_recognition_batch([], expected_count=0)
        payload["model_selection"] = selected.public_dict()
        return payload
    batch_size = max(1, min(len(image_paths), int(os.getenv("REC_BATCH_SIZE", "8"))))
    output = get_recognizer(selected.version, selected.variant).predict(
        input=image_paths,
        batch_size=batch_size,
    )
    payload = adapt_text_recognition_batch(output, expected_count=len(image_paths))
    payload["model_selection"] = selected.public_dict()
    return payload
