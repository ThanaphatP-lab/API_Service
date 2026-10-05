"""Typed, injectable transport boundary for pipeline-to-model calls."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from shared import upstream

ModelEndpoint = Literal[
    "/api/v1/layout-predictions",
    "/api/v1/layout-prediction-batches",
    "/api/v1/image-classification-batches",
    "/api/v1/text-detections",
    "/api/v1/text-detection-batches",
    "/api/v1/text-recognitions",
    "/api/v1/text-recognition-batches",
    "/api/v1/table-structures",
    "/api/v1/table-structure-batches",
    "/api/v1/ocr-results",
    "/api/v1/image-classifications",
]


class ModelClient(Protocol):
    def infer(
        self, base_url: str, endpoint: ModelEndpoint, image_paths: list[Path],
        *, request_id: str, fields: dict[str, Any] | None = None,
        multiple: bool = False,
    ) -> dict[str, Any]: ...


class HTTPModelClient:
    """Reuse existing authentication, timeouts, and upstream error translation.

    The client has no shared requests.Session, so concurrent pipeline calls do
    not share mutable transport state. A fake ModelClient can be injected in tests.
    """

    def infer(
        self, base_url: str, endpoint: ModelEndpoint, image_paths: list[Path],
        *, request_id: str, fields: dict[str, Any] | None = None,
        multiple: bool = False,
    ) -> dict[str, Any]:
        return upstream.post_images(
            base_url, endpoint, image_paths, request_id=request_id,
            fields=fields, multiple=multiple,
        )

    def readiness(self, base_url: str, *, request_id: str, timeout: float = 10.0) -> None:
        upstream.get_readiness(base_url, request_id=request_id, timeout=timeout)
