from __future__ import annotations

import os
from typing import Any

from paddleocr import TextDetection

from core.cache import singleflight_lru_cache
from shared.inference_adapters import adapt_text_detection, adapt_text_detection_batch
from shared.model_variants import ModelVariantSpec, resolve_model_variant
from shared.settings import device


def selection_from_settings(
    version: str | None = None,
    model: str = "baseline",
) -> ModelVariantSpec:
    return resolve_model_variant("detection", version, model)


@singleflight_lru_cache(maxsize=8)
def get_model(selection: ModelVariantSpec) -> TextDetection:
    options: dict[str, Any] = {
        "model_name": selection.model_name,
        "device": device(),
        "enable_mkldnn": False,
        "thresh": float(os.getenv("DET_THRESH", "0.25")),
        "box_thresh": float(os.getenv("DET_BOX_THRESH", "0.6")),
        "unclip_ratio": float(os.getenv("DET_UNCLIP_RATIO", "1.7")),
    }
    if selection.model_dir is not None:
        options["model_dir"] = str(selection.model_dir)
    return TextDetection(**options)


def infer(image_path: str, selection: ModelVariantSpec) -> dict[str, Any]:
    payload = adapt_text_detection(get_model(selection).predict(image_path))
    payload["model_selection"] = selection.public_dict()
    return payload


def infer_batch(
    image_paths: list[str],
    selection: ModelVariantSpec,
) -> dict[str, Any]:
    if not image_paths:
        payload = adapt_text_detection_batch([], expected_count=0)
        payload["model_selection"] = selection.public_dict()
        return payload
    try:
        configured_batch_size = int(os.getenv("DET_BATCH_SIZE", "1"))
    except ValueError:
        configured_batch_size = 8
    batch_size = max(1, min(len(image_paths), configured_batch_size))
    output = get_model(selection).predict(input=image_paths, batch_size=batch_size)
    payload = adapt_text_detection_batch(output, expected_count=len(image_paths))
    payload["model_selection"] = selection.public_dict()
    return payload


# Compatibility API retained while callers migrate from ``models.det_model``.
def get_detector(version: str | None = None, model: str = "baseline") -> TextDetection:
    return get_model(selection_from_settings(version, model))


get_detector.cache_clear = get_model.cache_clear  # type: ignore[attr-defined]
get_detector.cache_info = get_model.cache_info  # type: ignore[attr-defined]


def infer_detection(
    image_path: str,
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    return infer(image_path, selection_from_settings(version, model))


def infer_detection_batch(
    image_paths: list[str],
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    return infer_batch(image_paths, selection_from_settings(version, model))


def predict_detection(
    image_path: str,
    version: str | None = None,
    model: str = "baseline",
) -> list[dict[str, Any]]:
    return infer_detection(image_path, version, model)["predictions"]


def predict_detection_batch(
    image_paths: list[str],
    version: str | None = None,
    model: str = "baseline",
) -> list[dict[str, Any]]:
    return infer_detection_batch(image_paths, version, model)["predictions"]
