"""Gateway service inventory, independent of HTTP routing and model loading."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from shared.contracts import ModelAPIError


@dataclass(frozen=True)
class ServiceDescriptor:
    service_id: str
    kind: Literal["leaf", "pipeline"]
    routes: tuple[str, ...]


SERVICE_DESCRIPTORS = {
    "layout": ServiceDescriptor("pipeline-document-layout", "pipeline", ("/api/v1/document-layouts", "/api/v1/text-detections", "/api/v1/text-detection-batches")),
    "ocr-custom": ServiceDescriptor("pipeline-ocr-custom", "pipeline", ("/api/v1/ocr-results?engine=custom",)),
    "ocr-paddle": ServiceDescriptor("pipeline-ocr-paddle", "pipeline", ("/api/v1/ocr-results?engine=paddle", "/api/v1/ocr-result-batches")),
    "table": ServiceDescriptor("pipeline-table-custom", "pipeline", ("/api/v1/table-results",)),
    "table-model": ServiceDescriptor("pipeline-table-v2", "pipeline", ("/api/v1/table-model-results",)),
    "image-verification": ServiceDescriptor("pipeline-image-verification", "pipeline", ("/api/v1/image-verifications",)),
    "text-det-v5": ServiceDescriptor("leaf-text-detection-v5", "leaf", ("/api/v1/text-detections?version=v5", "/api/v1/text-detection-batches?version=v5")),
    "text-det-v6": ServiceDescriptor("leaf-text-detection-v6", "leaf", ("/api/v1/text-detections?version=v6", "/api/v1/text-detection-batches?version=v6")),
    "text-detection": ServiceDescriptor("leaf-text-detection", "leaf", ("/api/v1/text-detections?version=v5", "/api/v1/text-detections?version=v6", "/api/v1/text-detection-batches?version=v5", "/api/v1/text-detection-batches?version=v6")),
    "text-recognition": ServiceDescriptor("leaf-text-recognition", "leaf", ("/api/v1/text-recognitions", "/api/v1/text-recognition-batches")),
    "siglip": ServiceDescriptor("leaf-image-classification", "leaf", ("/api/v1/image-classifications",)),
}


def configured_services(variable: str, available: dict[str, str], *, default_all: bool = False) -> set[str]:
    # New SERVICES names take precedence, including an explicitly empty value.
    preferred = variable.replace("_PIPELINES", "_SERVICES")
    raw = os.getenv(preferred, os.getenv(variable, "all" if default_all else "")).strip().lower()
    if raw in {"all", "*"}:
        return set(available)
    names = {item.strip() for item in raw.split(",") if item.strip()}
    aliases = {value.service_id: key for key, value in SERVICE_DESCRIPTORS.items()}
    names = {aliases.get(name, name) for name in names}
    if "text-detection" in available:
        names = {"text-detection" if name in {"text-det-v5", "text-det-v6"} else name for name in names}
    return names


def validate_services(available: dict[str, str], enabled: set[str], required: set[str]) -> None:
    unknown = sorted((enabled | required) - set(available))
    disabled = sorted(required - enabled)
    if unknown or disabled or not enabled:
        raise ModelAPIError(
            503, "GATEWAY_CONFIGURATION_ERROR",
            "Gateway pipeline readiness configuration is invalid.",
            details=[{"unknown_pipelines": unknown, "required_but_disabled": disabled,
                      "supported_upstreams": sorted(available)}],
        )


def capabilities(available: dict[str, str], enabled: set[str], required: set[str]) -> list[dict]:
    """Report configured capabilities, not liveness or validated model availability."""
    validate_services(available, enabled, required)
    return [
        {"name": name, "service_id": SERVICE_DESCRIPTORS[name].service_id,
         "kind": SERVICE_DESCRIPTORS[name].kind, "required": name in required,
         "routes": list(SERVICE_DESCRIPTORS[name].routes)}
        for name in available if name in enabled
    ]
