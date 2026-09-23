from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _category_enabled(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("category.enabled must be a boolean")


def siglip_prompts(value: Any) -> list[str]:
    """Normalize notebook (4) category objects to model prompt strings.

    Plain string lists remain valid for the existing verification pipeline.
    Threshold fields intentionally stay outside the leaf inference service.
    """
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError("categories contains invalid JSON") from exc
        else:
            value = [part.strip() for part in stripped.split(",") if part.strip()]

    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("categories must be an array")

    prompts: list[str] = []
    for index, category in enumerate(value):
        if isinstance(category, Mapping):
            if not _category_enabled(category.get("enabled", True)):
                continue
            prompt = str(category.get("prompt") or "").strip()
            if not prompt:
                raise ValueError(f"categories[{index}].prompt is required")
        else:
            prompt = str(category).strip()
            if not prompt:
                continue
        prompts.append(prompt)
    return prompts
