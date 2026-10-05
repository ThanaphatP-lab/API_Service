from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from shared.contracts import ModelAPIError


ModelKind = Literal["detection", "recognition"]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "model_variants.json"
_VARIANT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class ModelVariantSpec:
    kind: ModelKind
    version: str
    variant: str
    model_name: str
    model_dir: Path | None
    requires_local_weights: bool

    @property
    def response_model(self) -> str:
        return self.model_name if self.variant == "baseline" else f"{self.model_name}:{self.variant}"

    def public_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "variant": self.variant,
            "model_name": self.model_name,
            "local_weights": self.model_dir is not None,
        }


@dataclass(frozen=True)
class OCRModelPairSpec:
    """A validated detection/recognition pair for an integrated OCR pipeline."""

    version: str
    variant: str
    detection: ModelVariantSpec
    recognition: ModelVariantSpec

    def public_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "variant": self.variant,
            "detection": self.detection.public_dict(),
            "recognition": self.recognition.public_dict(),
        }


def normalize_model_version(value: Any, *, default: str | None = None) -> str:
    normalized = str(value or default or "").strip().lower()
    aliases = {"5": "v5", "v5": "v5", "6": "v6", "v6": "v6"}
    if normalized not in aliases:
        raise ModelAPIError(
            422,
            "UNSUPPORTED_MODEL_VERSION",
            "version must be 5, v5, 6, or v6.",
            details=[{"field": "version", "received": normalized or None}],
        )
    return aliases[normalized]


def normalize_model_variant(value: Any) -> str:
    normalized = str(value or "baseline").strip().lower()
    if not _VARIANT_RE.fullmatch(normalized):
        raise ModelAPIError(
            422,
            "INVALID_MODEL_VARIANT",
            "model must contain only letters, numbers, underscores, or hyphens.",
            details=[{"field": "model", "received": normalized}],
        )
    return normalized


def configured_default_version(kind: ModelKind) -> str:
    prefix = "DET" if kind == "detection" else "REC"
    configured = os.getenv(f"{prefix}_MODEL_VERSION", "").strip()
    if configured:
        return normalize_model_version(configured)
    model_name = os.getenv(f"{prefix}_MODEL_NAME", "").lower()
    return "v6" if "ocrv6" in model_name else "v5"


def _config_path() -> Path:
    configured = os.getenv("MODEL_VARIANTS_CONFIG", "").strip()
    return Path(configured).expanduser().resolve() if configured else _DEFAULT_CONFIG_PATH


@lru_cache(maxsize=4)
def _load_config(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelAPIError(
            503,
            "MODEL_VARIANT_CONFIG_UNAVAILABLE",
            "The model variant configuration is unavailable.",
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelAPIError(
            503,
            "MODEL_VARIANT_CONFIG_INVALID",
            "The model variant configuration is invalid.",
        ) from exc
    if not isinstance(value, dict):
        raise ModelAPIError(
            503,
            "MODEL_VARIANT_CONFIG_INVALID",
            "The model variant configuration must be a JSON object.",
        )
    return value


def clear_model_variant_config_cache() -> None:
    _load_config.cache_clear()


def _has_model_artifacts(path: Path) -> bool:
    try:
        return path.is_dir() and any(
            item.is_file() and item.name != ".gitkeep"
            for item in path.iterdir()
        )
    except OSError:
        return False


def _resolve_directory(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip())
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def resolve_model_variant(
    kind: ModelKind,
    version: Any = None,
    variant: Any = "baseline",
) -> ModelVariantSpec:
    canonical_version = normalize_model_version(version, default=configured_default_version(kind))
    canonical_variant = normalize_model_variant(variant)
    config = _load_config(str(_config_path()))
    kind_config = config.get(kind)
    version_config = kind_config.get(canonical_version) if isinstance(kind_config, dict) else None
    entry = version_config.get(canonical_variant) if isinstance(version_config, dict) else None
    if not isinstance(entry, dict):
        available = sorted(version_config) if isinstance(version_config, dict) else []
        raise ModelAPIError(
            422,
            "UNSUPPORTED_MODEL_VARIANT",
            f"Model variant '{canonical_variant}' is not supported for {kind} {canonical_version}.",
            details=[
                {
                    "field": "model",
                    "version": canonical_version,
                    "received": canonical_variant,
                    "available": available,
                }
            ],
        )

    model_name = str(entry.get("model_name") or "").strip()
    if not model_name:
        raise ModelAPIError(
            503,
            "MODEL_VARIANT_CONFIG_INVALID",
            "The selected model variant has no model_name.",
        )

    requires_local = bool(entry.get("requires_local_weights", canonical_variant != "baseline"))
    configured_dir = _resolve_directory(entry.get("model_dir"))

    prefix = "DET" if kind == "detection" else "REC"
    if canonical_variant == "baseline" and canonical_version == configured_default_version(kind):
        model_name = os.getenv(f"{prefix}_MODEL_NAME", model_name).strip() or model_name
        environment_dir = _resolve_directory(os.getenv(f"{prefix}_MODEL_DIR"))
        if environment_dir is not None:
            configured_dir = environment_dir

    usable_dir = configured_dir if configured_dir is not None and _has_model_artifacts(configured_dir) else None
    if requires_local and usable_dir is None:
        raise ModelAPIError(
            422,
            "MODEL_VARIANT_UNAVAILABLE",
            f"Model variant '{canonical_variant}' is configured but its inference weights are unavailable.",
            details=[
                {
                    "field": "model",
                    "version": canonical_version,
                    "received": canonical_variant,
                }
            ],
        )

    return ModelVariantSpec(
        kind=kind,
        version=canonical_version,
        variant=canonical_variant,
        model_name=model_name,
        model_dir=usable_dir,
        requires_local_weights=requires_local,
    )


def resolve_ocr_model_pair(
    version: Any = None,
    variant: Any = "baseline",
    *,
    default_version: str = "v5",
    detection_variant: Any = None,
    recognition_variant: Any = None,
    detection_version: Any = None,
    recognition_version: Any = None,
) -> OCRModelPairSpec:
    """Resolve matching DET and REC variants for an integrated OCR pipeline."""

    canonical_version = normalize_model_version(version, default=default_version)
    det_version = normalize_model_version(detection_version, default=canonical_version)
    rec_version = normalize_model_version(recognition_version, default=canonical_version)
    default_variant = normalize_model_variant(variant)
    selected_detection_variant = normalize_model_variant(
        detection_variant if detection_variant is not None else default_variant
    )
    selected_recognition_variant = normalize_model_variant(
        recognition_variant if recognition_variant is not None else default_variant
    )
    profile = (
        selected_detection_variant
        if selected_detection_variant == selected_recognition_variant
        else f"det-{selected_detection_variant}__rec-{selected_recognition_variant}"
    )
    return OCRModelPairSpec(
        version=det_version if det_version == rec_version else f"det-{det_version}__rec-{rec_version}",
        variant=profile,
        detection=resolve_model_variant(
            "detection",
            det_version,
            selected_detection_variant,
        ),
        recognition=resolve_model_variant(
            "recognition",
            rec_version,
            selected_recognition_variant,
        ),
    )
