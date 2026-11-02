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
    "layout": ServiceDescriptor("pipeline-document-layout", "pipeline", ("/paddle/api/v1/document-layouts", "/paddle/api/v1/document-layout-batches")),
    "layout-model": ServiceDescriptor("leaf-layout", "leaf", ("/paddle/api/v1/layout-prediction-batches",)),
    "ocr-custom": ServiceDescriptor("pipeline-ocr-custom", "pipeline", ("/paddle/api/v1/ocr-results?engine=custom",)),
    "ocr-paddle": ServiceDescriptor("pipeline-ocr-paddle", "pipeline", ("/paddle/api/v1/ocr-results?engine=paddle", "/paddle/api/v1/ocr-result-batches")),
    "table": ServiceDescriptor("pipeline-table-custom", "pipeline", ("/paddle/api/v1/table-results",)),
    "table-model": ServiceDescriptor("pipeline-table-v2", "pipeline", ("/paddle/api/v1/table-model-results",)),
    "image-verification": ServiceDescriptor("pipeline-image-verification", "pipeline", ("/paddle/api/v1/image-verifications",)),
    "text-det-v5": ServiceDescriptor("leaf-text-detection-v5", "leaf", ("/paddle/api/v1/text-detections?version=v5", "/paddle/api/v1/text-detection-batches?version=v5")),
    "text-det-v6": ServiceDescriptor("leaf-text-detection-v6", "leaf", ("/paddle/api/v1/text-detections?version=v6", "/paddle/api/v1/text-detection-batches?version=v6")),
    "text-detection": ServiceDescriptor("leaf-text-detection", "leaf", ("/paddle/api/v1/text-detections", "/paddle/api/v1/text-detection-batches", "/paddle/api/v1/text-detections?version=v5", "/paddle/api/v1/text-detections?version=v6", "/paddle/api/v1/text-detection-batches?version=v5", "/paddle/api/v1/text-detection-batches?version=v6")),
    "text-recognition": ServiceDescriptor("leaf-text-recognition", "leaf", ("/paddle/api/v1/text-recognitions", "/paddle/api/v1/text-recognition-batches")),
    "siglip": ServiceDescriptor("leaf-image-classification", "leaf", ("/paddle/api/v1/image-classifications", "/paddle/api/v1/image-classification-batches")),
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
