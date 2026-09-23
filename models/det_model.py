from __future__ import annotations

import os
from typing import Any

from paddleocr import TextDetection

from shared.api import singleflight_lru_cache
from shared.inference_adapters import adapt_text_detection, adapt_text_detection_batch
from shared.model_variants import resolve_model_variant
from shared.settings import device


@singleflight_lru_cache(maxsize=8)
def get_detector(version: str | None = None, model: str = "baseline") -> TextDetection:
    selected = resolve_model_variant("detection", version, model)
    kwargs: dict[str, Any] = {
        "model_name": selected.model_name,
        "device": device(),
        "enable_mkldnn": False,
        "thresh": float(os.getenv("DET_THRESH", "0.25")),
        "box_thresh": float(os.getenv("DET_BOX_THRESH", "0.6")),
        "unclip_ratio": float(os.getenv("DET_UNCLIP_RATIO", "1.7")),
    }
    if selected.model_dir is not None:
        kwargs["model_dir"] = str(selected.model_dir)
    return TextDetection(**kwargs)


def predict_detection(
    image_path: str,
    version: str | None = None,
    model: str = "baseline",
) -> list[dict[str, Any]]:
    """Compatibility interface used by the in-process OCR pipeline."""
    return infer_detection(image_path, version, model)["predictions"]


def infer_detection(
    image_path: str,
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    """Run only model inference and serialize it to the canonical leaf contract."""
    selected = resolve_model_variant("detection", version, model)
    payload = adapt_text_detection(
        get_detector(selected.version, selected.variant).predict(image_path)
    )
    payload["model_selection"] = selected.public_dict()
    return payload


def predict_detection_batch(
    image_paths: list[str],
    version: str | None = None,
    model: str = "baseline",
) -> list[dict[str, Any]]:
    """Compatibility interface preserving one prediction mapping per input image."""
    return infer_detection_batch(image_paths, version, model)["predictions"]


def infer_detection_batch(
    image_paths: list[str],
    version: str | None = None,
    model: str = "baseline",
) -> dict[str, Any]:
    selected = resolve_model_variant("detection", version, model)
    if not image_paths:
        payload = adapt_text_detection_batch([], expected_count=0)
        payload["model_selection"] = selected.public_dict()
        return payload
    batch_size = max(1, min(len(image_paths), int(os.getenv("DET_BATCH_SIZE", "8"))))
    output = get_detector(selected.version, selected.variant).predict(
        input=image_paths,
        batch_size=batch_size,
    )
    payload = adapt_text_detection_batch(output, expected_count=len(image_paths))
    payload["model_selection"] = selected.public_dict()
    return payload
