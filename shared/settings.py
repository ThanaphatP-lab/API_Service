from __future__ import annotations

import os
from pathlib import Path


def model_dir(variable: str) -> str | None:
    """Return a configured local directory only if it contains model files.

    An empty mounted weights directory should not prevent PaddleOCR from resolving
    its official model by name.
    """
    value = os.getenv(variable)
    if value and Path(value).is_dir():
        # `.gitkeep` preserves the empty directory in source control; it is not
        # a model artifact and must not switch inference into local-weight mode.
        has_model_file = any(item.name != ".gitkeep" for item in Path(value).iterdir())
        if has_model_file:
            return value
    return None


def device() -> str:
    return os.getenv("MODEL_DEVICE", "gpu:0")
