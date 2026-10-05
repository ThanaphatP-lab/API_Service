from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from paddleocr import LayoutDetection

from core.cache import singleflight_lru_cache
from shared.inference_adapters import adapt_layout
from shared.settings import device, model_dir
from core.settings import _positive_int_env


@dataclass(frozen=True)
class LayoutSelection:
    model_name: str
    model_dir: str | None
    device: str


def selection_from_settings() -> LayoutSelection:
    return LayoutSelection(
        model_name=os.getenv("LAYOUT_MODEL_NAME", "PP-DocLayoutV3"),
        model_dir=model_dir("LAYOUT_MODEL_DIR"),
        device=device(),
    )


@singleflight_lru_cache(maxsize=1)
def _load_model(selection: LayoutSelection) -> LayoutDetection:
    options: dict[str, Any] = {
        "model_name": selection.model_name,
        "device": selection.device,
        "enable_mkldnn": False,
    }
    if selection.model_dir is not None:
        options["model_dir"] = selection.model_dir
    return LayoutDetection(**options)


def get_model(selection: LayoutSelection | None = None) -> LayoutDetection:
    return _load_model(selection or selection_from_settings())


def infer(image_path: str, selection: LayoutSelection | None = None) -> dict[str, Any]:
    selected = selection or selection_from_settings()
    return adapt_layout(get_model(selected).predict(image_path))


def infer_batch(image_paths: list[str], selection: LayoutSelection | None = None) -> dict[str, Any]:
    results = []
    if image_paths:
        model = get_model(selection or selection_from_settings())
        batch_size = min(len(image_paths), _positive_int_env("LAYOUT_BATCH_SIZE", 1))
        for prediction in model.predict(input=image_paths, batch_size=batch_size):
            results.append(adapt_layout([prediction]))
        if len(results) != len(image_paths):
            raise RuntimeError("Layout result count does not match input count")
    return {"results": results, "count": len(results)}
