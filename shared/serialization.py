from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

try:
    import numpy as np
except ImportError:  # API-only services do not need NumPy installed.
    np = None  # type: ignore[assignment]


def to_jsonable(value: Any) -> Any:
    """Convert PaddleOCR/PyTorch result objects and NumPy values to JSON values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if np is not None and isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if np is not None and isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    # Paddle result objects commonly expose their useful mapping through res.
    try:
        result = getattr(value, "res", None)
        if result is not None:
            return to_jsonable(result)
    except Exception:
        pass
    for attribute in ("numpy", "tolist"):
        try:
            converter = getattr(value, attribute, None)
            if callable(converter):
                return to_jsonable(converter())
        except Exception:
            pass
    if hasattr(value, "json"):
        raw = value.json
        raw = raw() if callable(raw) else raw
        if isinstance(raw, str):
            return to_jsonable(json.loads(raw))
        return to_jsonable(raw)
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    if hasattr(value, "item") and callable(value.item):
        try:
            return to_jsonable(value.item())
        except (ValueError, RuntimeError):
            pass
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    # Keep the API response valid JSON even for third-party result classes that
    # expose none of the common conversion hooks above.
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def prediction_list(predictions: Any) -> list[Any]:
    if predictions is None:
        return []
    if isinstance(predictions, list):
        values = predictions
    elif isinstance(predictions, tuple):
        values = list(predictions)
    elif isinstance(predictions, Mapping):
        values = [predictions]
    else:
        try:
            values = list(predictions)
        except TypeError:
            values = [predictions]
    return [to_jsonable(prediction) for prediction in values]
