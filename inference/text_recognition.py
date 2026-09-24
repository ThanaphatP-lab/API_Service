from __future__ import annotations

import logging
import os
from typing import Any

from paddleocr import TextRecognition

from core.cache import singleflight_lru_cache
from shared.inference_adapters import adapt_text_recognition, adapt_text_recognition_batch
from shared.model_variants import ModelVariantSpec, resolve_model_variant
from shared.settings import device


logger = logging.getLogger("uvicorn.error")


def selection_from_settings(
    version: str | None = None,
    model: str = "baseline",
) -> ModelVariantSpec:
    return resolve_model_variant("recognition", version, model)


@singleflight_lru_cache(maxsize=8)
def get_model(selection: ModelVariantSpec) -> TextRecognition:
    selected_device = device()
    options: dict[str, Any] = {
        "model_name": selection.model_name,
        "device": selected_device,
        "enable_mkldnn": False,
    }
    if selection.model_dir is not None:
        options["model_dir"] = str(selection.model_dir)
    logger.info(
        "Loading recognition model version=%s variant=%s model_name=%s model_dir=%s device=%s",
        selection.version,
        selection.variant,
        selection.model_name,
        str(selection.model_dir) if selection.model_dir is not None else "<official-model-cache>",
        selected_device,
    )
    recognizer = TextRecognition(**options)
    logger.info(
        "Recognition model loaded version=%s variant=%s model_name=%s local_weights=%s",
        selection.version,
        selection.variant,
        selection.model_name,
        selection.model_dir is not None,
    )
    return recognizer


def infer(image_path: str, selection: ModelVariantSpec) -> dict[str, Any]:
    payload = adapt_text_recognition(get_model(selection).predict(image_path))
    payload["model_selection"] = selection.public_dict()
    return payload


def infer_batch(
    image_paths: list[str],
    selection: ModelVariantSpec,
) -> dict[str, Any]:
    if not image_paths:
        payload = adapt_text_recognition_batch([], expected_count=0)
        payload["model_selection"] = selection.public_dict()
        return payload
    try:
        configured_batch_size = int(os.getenv("REC_BATCH_SIZE", "8"))
    except ValueError:
        configured_batch_size = 8
    batch_size = max(1, min(len(image_paths), configured_batch_size))
    output = get_model(selection).predict(input=image_paths, batch_size=batch_size)
    payload = adapt_text_recognition_batch(output, expected_count=len(image_paths))
    payload["model_selection"] = selection.public_dict()
    return payload


# Compatibility API retained while callers migrate from ``models.rec_model``.
def get_recognizer(version: str | None = None, model: str = "baseline") -> TextRecognition:
    return get_model(selection_from_settings(version, model))


get_recognizer.cache_clear = get_model.cache_clear  # type: ignore[attr-defined]
get_recognizer.cache_info = get_model.cache_info  # type: ignore[attr-defined]


def infer_recognition(
    image_path: str,
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    return infer(image_path, selection_from_settings(version, model))


def infer_recognition_batch(
    image_paths: list[str],
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    return infer_batch(image_paths, selection_from_settings(version, model))


def predict_recognition(
    image_path: str,
    version: str | None = None,
    model: str = "baseline",
) -> list[dict[str, Any]]:
    return infer_recognition(image_path, version, model)["predictions"]


def predict_recognition_batch(
    image_paths: list[str],
    version: str | None = None,
    model: str = "baseline",
) -> list[dict[str, Any]]:
    return infer_recognition_batch(image_paths, version, model)["predictions"]
