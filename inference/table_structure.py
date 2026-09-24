from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from paddleocr import TableStructureRecognition

from core.cache import singleflight_lru_cache
from shared.inference_adapters import adapt_table_structure
from shared.settings import device, model_dir


@dataclass(frozen=True)
class TableStructureSelection:
    model_name: str
    model_dir: str | None
    device: str


def selection_from_settings() -> TableStructureSelection:
    return TableStructureSelection(
        model_name=os.getenv("TABLE_MODEL_NAME", "SLANeXt_wired"),
        model_dir=model_dir("TABLE_MODEL_DIR"),
        device=device(),
    )


@singleflight_lru_cache(maxsize=1)
def _load_model(selection: TableStructureSelection) -> TableStructureRecognition:
    options: dict[str, Any] = {
        "model_name": selection.model_name,
        "device": selection.device,
        "enable_mkldnn": False,
    }
    if selection.model_dir is not None:
        options["model_dir"] = selection.model_dir
    return TableStructureRecognition(**options)


def get_model(selection: TableStructureSelection | None = None) -> TableStructureRecognition:
    return _load_model(selection or selection_from_settings())


def infer(image_path: str, selection: TableStructureSelection | None = None) -> dict[str, Any]:
    selected = selection or selection_from_settings()
    return adapt_table_structure(get_model(selected).predict(image_path))


def infer_batch(
    image_paths: list[str],
    selection: TableStructureSelection | None = None,
) -> dict[str, Any]:
    selected = selection or selection_from_settings()
    if not image_paths:
        payload = adapt_table_structure([])
        payload["count"] = 0
        return payload
    try:
        configured_batch_size = int(os.getenv("TABLE_BATCH_SIZE", "4"))
    except ValueError:
        configured_batch_size = 4
    batch_size = max(1, min(len(image_paths), configured_batch_size))
    payload = adapt_table_structure(
        get_model(selected).predict(input=image_paths, batch_size=batch_size)
    )
    payload["count"] = len(image_paths)
    return payload
