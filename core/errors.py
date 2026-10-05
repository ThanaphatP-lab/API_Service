from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from fastapi import Request
from shared.contracts import ModelAPIError, request_id
from core.settings import runtime_settings
from core.telemetry import timed_stage


_SENSITIVE_DETAIL_KEYS = {"path", "reason", "stack", "traceback", "upstream", "upstream_message"}


def _safe_received(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value if not isinstance(value, str) else value[:200]
    if isinstance(value, list):
        return [_safe_received(item) for item in value[:10]]
    if isinstance(value, dict):
        return {str(key): _safe_received(item) for key, item in list(value.items())[:10]}
    return str(value)[:200]


def _public_error_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove internal topology and exception text from public API responses."""

    def sanitize(value: Any, key: str = "") -> Any:
        if key.lower() in _SENSITIVE_DETAIL_KEYS:
            return None
        if isinstance(value, dict):
            return {
                str(child_key): sanitized
                for child_key, child in value.items()
                if (sanitized := sanitize(child, str(child_key))) is not None
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value[:20]]
        return _safe_received(value)

    return [sanitize(detail) for detail in details[:20]]


def _docs_url(request: Request, code: str) -> str:
    configured = runtime_settings.error_docs_url
    if configured:
        return f"{configured.rstrip('/')}#{code.lower()}"
    return f"{str(request.base_url).rstrip('/')}/docs"


def _error_body(
    request: Request,
    *,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or [],
            "request_id": request_id(request),
            "docs": _docs_url(request, code),
        }
    }


@timed_stage("inference_including_load_and_adaptation")
def run_image_inference(handler: Callable[[str], Any], path: Path) -> Any:
    try:
        return handler(str(path))
    except ModelAPIError:
        raise
    except FileNotFoundError as exc:
        raise ModelAPIError(
            503,
            "MODEL_WEIGHTS_UNAVAILABLE",
            "Model weights are unavailable.",
            details=[{"reason": str(exc)[:500]}],
        ) from exc
    except (RuntimeError, OSError) as exc:
        message = str(exc)
        lowered = message.lower()
        code = "GPU_UNAVAILABLE" if any(token in lowered for token in ("cuda", "gpu", "cudnn")) else "MODEL_INFERENCE_FAILED"
        status = 503 if code == "GPU_UNAVAILABLE" else 500
        raise ModelAPIError(
            status,
            code,
            "The model could not complete inference.",
            details=[{"reason": message[:500]}],
        ) from exc
    except Exception as exc:
        raise ModelAPIError(
            500,
            "MODEL_INFERENCE_FAILED",
            "The model could not complete inference.",
            details=[{"reason": str(exc)[:500]}],
        ) from exc

