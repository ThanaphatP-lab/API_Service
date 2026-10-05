from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from shared.contracts import ModelAPIError
from core.telemetry import timed_stage


def chunked_paths(image_paths: list[Path], batch_size: int) -> list[list[Path]]:
    """Split paths into stable, ordered batches for an upstream receiver.

    ``MAX_BATCH_IMAGES`` protects the receiving service; it does not split a
    caller's request.  Composition pipelines use this helper so a page with
    more text polygons than the leaf limit keeps its original result order.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [image_paths[index : index + batch_size] for index in range(0, len(image_paths), batch_size)]


def _internal_headers(request_id: str) -> dict[str, str]:
    headers = {"X-Request-ID": request_id}
    token = os.getenv("INTERNAL_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@timed_stage("upstream_http_roundtrip")
def post_images(
    base_url: str,
    endpoint: str,
    image_paths: list[Path],
    *,
    fields: dict[str, Any] | None = None,
    request_id: str,
    timeout: float = 240.0,
    multiple: bool = False,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    handles = []
    try:
        files = []
        field_name = "images" if multiple else "image"
        for path in image_paths:
            handle = path.open("rb")
            handles.append(handle)
            files.append((field_name, (path.name, handle, "application/octet-stream")))
        form = {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            for key, value in (fields or {}).items()
            if value is not None
        }
        response = requests.post(
            url,
            files=files,
            data=form,
            headers=_internal_headers(request_id),
            timeout=timeout,
        )
    except requests.Timeout as exc:
        raise ModelAPIError(
            504,
            "UPSTREAM_TIMEOUT",
            "An upstream model service timed out.",
            details=[{"upstream": url, "timeout_seconds": timeout}],
        ) from exc
    except requests.RequestException as exc:
        raise ModelAPIError(
            503,
            "UPSTREAM_UNAVAILABLE",
            "An upstream model service is unavailable.",
            details=[{"upstream": url, "reason": str(exc)[:500]}],
        ) from exc
    finally:
        for handle in handles:
            handle.close()

    try:
        body = response.json()
    except ValueError as exc:
        raise ModelAPIError(
            502,
            "UPSTREAM_INVALID_RESPONSE",
            "An upstream model service returned invalid JSON.",
            details=[{"upstream": url, "status": response.status_code}],
        ) from exc

    if not response.ok:
        upstream_error = body.get("error", {}) if isinstance(body, dict) else {}
        raise ModelAPIError(
            response.status_code if 400 <= response.status_code < 500 else 502,
            "UPSTREAM_MODEL_ERROR",
            "An upstream model service rejected or failed the request.",
            details=[
                {
                    "upstream": url,
                    "upstream_status": response.status_code,
                    "upstream_code": upstream_error.get("code"),
                    "upstream_message": upstream_error.get("message"),
                    "upstream_request_id": upstream_error.get("request_id") or response.headers.get("X-Request-ID"),
                }
            ],
        )
    if not isinstance(body, dict) or "data" not in body:
        raise ModelAPIError(
            502,
            "UPSTREAM_CONTRACT_MISMATCH",
            "An upstream model service did not follow the v1 response contract.",
            details=[{"upstream": url}],
        )
    return body["data"]


def get_readiness(base_url: str, *, request_id: str, timeout: float = 10.0) -> None:
    url = f"{base_url.rstrip('/')}/api/v1/readiness"
    try:
        response = requests.get(url, headers=_internal_headers(request_id), timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ModelAPIError(
            503,
            "UPSTREAM_NOT_READY",
            "A required upstream model service is not ready.",
            details=[{"upstream": url, "reason": str(exc)[:500]}],
        ) from exc
