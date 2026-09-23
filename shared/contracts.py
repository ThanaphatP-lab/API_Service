from __future__ import annotations

import os
import time
from typing import Any

from fastapi import Request


API_VERSION = "v1"


class ModelAPIError(RuntimeError):
    """An expected API failure with a stable, machine-readable error code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []
        self.headers = headers or {}


def request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def elapsed_ms(request: Request) -> float:
    started = float(getattr(request.state, "started_at", time.perf_counter()))
    return round((time.perf_counter() - started) * 1000, 2)


def response_meta(
    request: Request,
    *,
    service: str,
    model: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "request_id": request_id(request),
        "api_version": API_VERSION,
        "service": service,
        "duration_ms": elapsed_ms(request),
    }
    if model:
        meta["model"] = model
    version = model_version or os.getenv("MODEL_VERSION", "").strip()
    if version:
        meta["model_version"] = version
    return meta


def success_response(
    request: Request,
    data: Any,
    *,
    service: str,
    model: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    return {
        "data": data,
        "meta": response_meta(
            request,
            service=service,
            model=model,
            model_version=model_version,
        ),
    }
