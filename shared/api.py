from __future__ import annotations

import asyncio
import base64
import binascii
import hmac
import io
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from shared.contracts import API_VERSION, ModelAPIError, request_id, success_response
from shared.rate_limit import InMemoryRateLimiter, client_key, rate_limit_headers


logger = logging.getLogger("model_api")
_AUTH_EXEMPT_PATHS = {"/health", f"/api/{API_VERSION}/health"}
_SENSITIVE_DETAIL_KEYS = {"path", "reason", "stack", "traceback", "upstream", "upstream_message"}


def singleflight_lru_cache(maxsize: int = 1):
    """Cache a loader while allowing only one concurrent cache miss."""

    def decorate(loader: Callable[..., Any]) -> Callable[..., Any]:
        cached = lru_cache(maxsize=maxsize)(loader)
        lock = RLock()

        @wraps(loader)
        def synchronized(*args: Any, **kwargs: Any) -> Any:
            # lru_cache protects its dictionary, but it does not prevent two
            # threads from executing the same uncached GPU loader together.
            with lock:
                return cached(*args, **kwargs)

        synchronized.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
        synchronized.cache_info = cached.cache_info  # type: ignore[attr-defined]
        return synchronized

    return decorate


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _is_production() -> bool:
    return os.getenv("APP_ENV", "development").strip().lower() in {"prod", "production"}


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


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


def _valid_configured_token(variable: str, token: str) -> None:
    looks_like_placeholder = token.lower().startswith(("replace-", "change-me", "changeme")) or (
        token.startswith("<") and token.endswith(">")
    )
    if _is_production() and (len(token) < 32 or looks_like_placeholder):
        raise RuntimeError(
            f"{variable} must contain a non-placeholder secret of at least 32 characters "
            "when APP_ENV=production."
        )


def _configured_tokens(variable: str) -> tuple[str, ...]:
    primary = os.getenv(variable, "").strip()
    _valid_configured_token(variable, primary)
    if not primary:
        return ()
    previous_variable = f"{variable}_PREVIOUS"
    previous = os.getenv(previous_variable, "").strip()
    if previous:
        _valid_configured_token(previous_variable, previous)
    return tuple(dict.fromkeys(token for token in (primary, previous) if token))


def _request_token(request: Request) -> str:
    authorization = request.headers.get("authorization", "").strip()
    if authorization:
        scheme, separator, credentials = authorization.partition(" ")
        if separator and scheme.lower() == "bearer":
            return credentials.strip()
    return request.headers.get("x-api-key", "").strip()


def _security_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Content-Security-Policy": "frame-ancestors 'none'",
    }
    if _is_production():
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if extra:
        headers.update(extra)
    return headers

IMAGE_REQUEST_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "format": "binary"},
                        "image_base64": {"type": "string", "description": "Optional Data URL/Base64 alternative"},
                    },
                }
            },
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["image"],
                    "properties": {"image": {"type": "string", "description": "Base64 or Data URL"}},
                    "additionalProperties": True,
                }
            },
        },
    }
}

BATCH_IMAGE_REQUEST_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "images": {"type": "array", "items": {"type": "string", "format": "binary"}},
                    },
                }
            },
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["images"],
                    "properties": {"images": {"type": "array", "items": {"type": "string"}}},
                }
            },
        },
    }
}


@dataclass
class ImageRequest:
    paths: list[Path]
    fields: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.paths[0]

    def cleanup(self) -> None:
        for path in self.paths:
            path.unlink(missing_ok=True)


def _docs_url(request: Request, code: str) -> str:
    configured = os.getenv("ERROR_DOCS_URL", "").strip()
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


def _safe_received(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value if not isinstance(value, str) else value[:200]
    if isinstance(value, list):
        return [_safe_received(item) for item in value[:10]]
    if isinstance(value, dict):
        return {str(key): _safe_received(item) for key, item in list(value.items())[:10]}
    return str(value)[:200]


def create_app(
    title: str,
    model_name: str,
    *,
    service_name: str | None = None,
    auth_token_env: str = "INTERNAL_API_TOKEN",
    required_auth_token_envs: tuple[str, ...] = (),
    public_api: bool = False,
) -> FastAPI:
    service = service_name or title.lower().replace(" ", "-")
    docs_enabled = not _is_production() and _env_flag("API_DOCS_ENABLED", "true")
    app = FastAPI(
        title=title,
        version="1.0.0",
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.state.service_name = service
    app.state.model_name = model_name
    app.state.public_api = public_api
    expected_tokens = _configured_tokens(auth_token_env)
    for required_token_env in required_auth_token_envs:
        _configured_tokens(required_token_env)

    if docs_enabled:
        def secured_openapi() -> dict[str, Any]:
            if app.openapi_schema:
                return app.openapi_schema
            schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
            schemes = schema.setdefault("components", {}).setdefault("securitySchemes", {})
            schemes["BearerAuth"] = {"type": "http", "scheme": "bearer"}
            schemes["ApiKeyAuth"] = {"type": "apiKey", "in": "header", "name": "X-API-Key"}
            schema["security"] = [{"BearerAuth": []}, {"ApiKeyAuth": []}]
            for health_path in ("/health", f"/api/{API_VERSION}/health"):
                for operation in schema.get("paths", {}).get(health_path, {}).values():
                    if isinstance(operation, dict):
                        operation["security"] = []
            app.openapi_schema = schema
            return schema

        app.openapi = secured_openapi
    origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "").split(",") if item.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials="*" not in origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Request-ID"],
            expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"],
        )
    limiter = InMemoryRateLimiter()
    max_concurrent = _positive_int_env("MAX_CONCURRENT_REQUESTS", 2 if public_api else 1)
    queue_timeout = float(os.getenv("INFERENCE_QUEUE_TIMEOUT_SECONDS", "5"))
    inference_slots = asyncio.Semaphore(max_concurrent)

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[..., Any]):
        supplied = request.headers.get("x-request-id", "").strip()
        valid_supplied = bool(
            supplied
            and len(supplied) <= 128
            and supplied.replace("-", "").replace("_", "").isalnum()
        )
        request.state.request_id = supplied if valid_supplied else f"req_{uuid.uuid4().hex}"
        request.state.started_at = time.perf_counter()
        documentation_path = docs_enabled and request.url.path in {"/docs", "/redoc", "/openapi.json"}

        if _is_production() and request.url.path == "/predict" and not _env_flag("LEGACY_ENDPOINTS_ENABLED"):
            return JSONResponse(
                status_code=404,
                content=_error_body(request, code="RESOURCE_NOT_FOUND", message="The requested resource was not found."),
                headers=_security_headers({"X-Request-ID": request.state.request_id}),
            )

        decision = None
        if request.url.path not in _AUTH_EXEMPT_PATHS:
            decision = limiter.check(client_key(request))
            if not decision.allowed:
                headers = rate_limit_headers(decision)
                headers["Retry-After"] = str(decision.reset_after)
                headers["X-Request-ID"] = request.state.request_id
                return JSONResponse(
                    status_code=429,
                    content=_error_body(
                        request,
                        code="RATE_LIMIT_EXCEEDED",
                        message="Too many requests. Retry after the indicated interval.",
                        details=[{"retry_after_seconds": decision.reset_after}],
                    ),
                    headers=_security_headers(headers),
                )

        if (
            expected_tokens
            and request.method != "OPTIONS"
            and request.url.path not in _AUTH_EXEMPT_PATHS
            and not documentation_path
        ):
            provided_token = _request_token(request)
            if not provided_token or not any(
                hmac.compare_digest(provided_token, expected_token) for expected_token in expected_tokens
            ):
                logger.warning(
                    "authentication_failed request_id=%s service=%s client=%s",
                    request.state.request_id,
                    service,
                    client_key(request),
                )
                return JSONResponse(
                    status_code=401,
                    content=_error_body(
                        request,
                        code="AUTHENTICATION_REQUIRED",
                        message="A valid Bearer token is required.",
                    ),
                    headers=_security_headers({
                        "WWW-Authenticate": "Bearer",
                        "X-Request-ID": request.state.request_id,
                    }),
                )

        content_length = request.headers.get("content-length", "").strip()
        if content_length:
            try:
                received_bytes = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content=_error_body(request, code="INVALID_CONTENT_LENGTH", message="Content-Length must be numeric."),
                    headers=_security_headers({"X-Request-ID": request.state.request_id}),
                )
            max_request_bytes = _positive_int_env("MAX_REQUEST_MB", 28) * 1024 * 1024
            if received_bytes > max_request_bytes:
                return JSONResponse(
                    status_code=413,
                    content=_error_body(
                        request,
                        code="PAYLOAD_TOO_LARGE",
                        message="The request body exceeds the configured size limit.",
                        details=[{"received_bytes": received_bytes, "max_bytes": max_request_bytes}],
                    ),
                    headers=_security_headers({"X-Request-ID": request.state.request_id}),
                )

        acquired = False
        if request.method == "POST":
            try:
                await asyncio.wait_for(inference_slots.acquire(), timeout=max(0.1, queue_timeout))
                acquired = True
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=503,
                    content=_error_body(
                        request,
                        code="SERVICE_BUSY",
                        message="The inference queue is full. Retry later.",
                        details=[{"retry_after_seconds": 1}],
                    ),
                    headers=_security_headers({"Retry-After": "1", "X-Request-ID": request.state.request_id}),
                )
        try:
            response = await call_next(request)
        finally:
            if acquired:
                inference_slots.release()
        response.headers["X-Request-ID"] = request.state.request_id
        for key, value in _security_headers().items():
            response.headers[key] = value
        if decision is not None:
            for key, value in rate_limit_headers(decision).items():
                response.headers[key] = value
        elapsed = round((time.perf_counter() - request.state.started_at) * 1000, 2)
        response.headers["Server-Timing"] = f"app;dur={elapsed}"
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%s",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    @app.exception_handler(ModelAPIError)
    async def model_api_error_handler(request: Request, exc: ModelAPIError) -> JSONResponse:
        logger.warning(
            "model_api_error request_id=%s code=%s status=%s details=%r",
            request_id(request),
            exc.code,
            exc.status_code,
            exc.details,
        )
        details = exc.details
        if public_api and not _env_flag("DEBUG_ERRORS"):
            details = _public_error_details(details)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                request,
                code=exc.code,
                message=exc.message,
                details=details,
            ),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in error.get("loc", []) if part not in {"body", "query"}),
                "issue": error.get("msg", "Invalid value"),
                "type": error.get("type", "validation_error"),
                "received": _safe_received(error.get("input")),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_error_body(
                request,
                code="VALIDATION_ERROR",
                message="The request did not pass validation.",
                details=details,
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        code_by_status = {
            400: "BAD_REQUEST",
            404: "RESOURCE_NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "PAYLOAD_TOO_LARGE",
            415: "UNSUPPORTED_MEDIA_TYPE",
            422: "VALIDATION_ERROR",
            503: "SERVICE_UNAVAILABLE",
        }
        detail = exc.detail
        details = detail if isinstance(detail, list) else []
        message = detail if isinstance(detail, str) else "The request could not be processed."
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(
                request,
                code=code_by_status.get(exc.status_code, "HTTP_ERROR"),
                message=message,
                details=details,
            ),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error request_id=%s", request_id(request), exc_info=exc)
        details: list[dict[str, Any]] = []
        if os.getenv("DEBUG_ERRORS", "false").strip().lower() in {"1", "true", "yes", "on"}:
            details.append({"exception_type": type(exc).__name__, "reason": str(exc)[:500]})
        return JSONResponse(
            status_code=500,
            content=_error_body(
                request,
                code="INTERNAL_ERROR",
                message="An unexpected error occurred. Use request_id to inspect server logs.",
                details=details,
            ),
        )

    @app.get(f"/api/{API_VERSION}/health", tags=["Operations"])
    def health(request: Request) -> dict[str, Any]:
        return success_response(
            request,
            {
                "status": "ok",
                "model": model_name,
                "device": os.getenv("MODEL_DEVICE", "gpu:0"),
            },
            service=service,
            model=model_name,
        )

    @app.get("/health", include_in_schema=False)
    def legacy_health(request: Request) -> dict[str, Any]:
        return health(request)

    return app


def add_readiness_route(app: FastAPI, loader: Callable[[], Any]) -> None:
    service = str(app.state.service_name)
    model_name = str(app.state.model_name)

    @app.get(f"/api/{API_VERSION}/readiness", tags=["Operations"])
    def readiness(request: Request) -> dict[str, Any]:
        try:
            loader()
        except Exception as exc:
            logger.exception("Model readiness failed request_id=%s", request_id(request), exc_info=exc)
            raise ModelAPIError(
                503,
                "MODEL_NOT_READY",
                "The model could not be loaded.",
                details=[{"model": model_name, "reason": str(exc)[:500]}],
            ) from exc
        return success_response(
            request,
            {"status": "ready"},
            service=service,
            model=model_name,
        )


def _max_upload_bytes() -> int:
    return max(1, int(os.getenv("MAX_UPLOAD_MB", "20"))) * 1024 * 1024


def _decode_base64(value: str) -> bytes:
    encoded = value.split(",", 1)[1] if "," in value else value
    max_encoded_length = ((_max_upload_bytes() + 2) // 3) * 4 + 4
    if len(encoded) > max_encoded_length:
        raise ModelAPIError(
            413,
            "PAYLOAD_TOO_LARGE",
            "The Base64 image exceeds the configured image size limit.",
            details=[{"max_bytes": _max_upload_bytes()}],
        )
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ModelAPIError(
            422,
            "INVALID_IMAGE_BASE64",
            "The image field is not valid Base64.",
            details=[{"field": "image", "issue": "invalid_base64"}],
        ) from exc


def _write_verified_image(data: bytes) -> Path:
    if not data:
        raise ModelAPIError(422, "EMPTY_IMAGE", "The uploaded image is empty.")
    if len(data) > _max_upload_bytes():
        raise ModelAPIError(
            413,
            "PAYLOAD_TOO_LARGE",
            f"Image exceeds the {int(os.getenv('MAX_UPLOAD_MB', '20'))} MB limit.",
            details=[{"received_bytes": len(data), "max_bytes": _max_upload_bytes()}],
        )
    try:
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            max_pixels = _positive_int_env("MAX_IMAGE_PIXELS", 40_000_000)
            max_dimension = _positive_int_env("MAX_IMAGE_DIMENSION", 12_000)
            if width <= 0 or height <= 0 or width > max_dimension or height > max_dimension or width * height > max_pixels:
                raise ModelAPIError(
                    413,
                    "IMAGE_DIMENSIONS_TOO_LARGE",
                    "The image dimensions exceed the configured limit.",
                    details=[
                        {
                            "width": width,
                            "height": height,
                            "max_dimension": max_dimension,
                            "max_pixels": max_pixels,
                        }
                    ],
                )
            source.verify()
            image_format = (source.format or "PNG").lower()
    except (UnidentifiedImageError, OSError) as exc:
        raise ModelAPIError(
            422,
            "INVALID_IMAGE",
            "The uploaded content is not a readable image.",
            details=[{"field": "image", "issue": "decode_failed"}],
        ) from exc
    suffix = ".jpg" if image_format in {"jpg", "jpeg"} else f".{image_format}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        return Path(handle.name)


async def _read_upload_limited(upload: Any) -> bytes:
    max_bytes = _max_upload_bytes()
    chunks: list[bytes] = []
    received = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        received += len(chunk)
        if received > max_bytes:
            raise ModelAPIError(
                413,
                "PAYLOAD_TOO_LARGE",
                "An uploaded image exceeds the configured image size limit.",
                details=[{"received_bytes": received, "max_bytes": max_bytes}],
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def parse_image_request(request: Request, *, multiple: bool = False) -> ImageRequest:
    """Accept image input as multipart upload or JSON Base64/Data URL."""
    content_type = request.headers.get("content-type", "").lower()
    fields: dict[str, Any] = {}
    raw_images: list[bytes] = []

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        for key, value in form.multi_items():
            if hasattr(value, "read"):
                if key in {"image", "images"}:
                    raw_images.append(await _read_upload_limited(value))
                continue
            if key in fields:
                existing = fields[key]
                fields[key] = [existing, value] if not isinstance(existing, list) else [*existing, value]
            else:
                fields[key] = value
        base64_value = fields.pop("image_base64", None)
        if base64_value and not raw_images:
            values = base64_value if isinstance(base64_value, list) else [base64_value]
            raw_images.extend(_decode_base64(str(value)) for value in values)
    elif content_type.startswith("application/json"):
        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ModelAPIError(400, "INVALID_JSON", "The request body is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ModelAPIError(422, "VALIDATION_ERROR", "The JSON body must be an object.")
        values = payload.get("images") if multiple else payload.get("image", payload.get("image_base64"))
        if values is not None:
            image_values = values if isinstance(values, list) else [values]
            raw_images.extend(_decode_base64(str(value)) for value in image_values)
        fields = {key: value for key, value in payload.items() if key not in {"image", "images", "image_base64"}}
    else:
        raise ModelAPIError(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            "Use multipart/form-data or application/json with a Base64 image.",
            details=[{"received": content_type or "missing"}],
        )

    if not raw_images:
        field_name = "images" if multiple else "image"
        raise ModelAPIError(
            422,
            "IMAGE_REQUIRED",
            f"The {field_name} field is required.",
            details=[{"field": field_name, "issue": "required"}],
        )
    if not multiple and len(raw_images) != 1:
        raise ModelAPIError(422, "TOO_MANY_IMAGES", "This endpoint accepts exactly one image.")
    if multiple:
        max_images = _positive_int_env("MAX_BATCH_IMAGES", 64)
        if len(raw_images) > max_images:
            raise ModelAPIError(
                413,
                "BATCH_TOO_LARGE",
                "The image batch contains too many items.",
                details=[{"received_images": len(raw_images), "max_images": max_images}],
            )
        max_batch_bytes = _positive_int_env("MAX_BATCH_UPLOAD_MB", 64) * 1024 * 1024
        total_bytes = sum(len(data) for data in raw_images)
        if total_bytes > max_batch_bytes:
            raise ModelAPIError(
                413,
                "BATCH_TOO_LARGE",
                "The image batch exceeds the configured total size limit.",
                details=[{"received_bytes": total_bytes, "max_bytes": max_batch_bytes}],
            )
    paths: list[Path] = []
    try:
        paths = [_write_verified_image(data) for data in raw_images]
        return ImageRequest(paths=paths, fields=fields)
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        raise


def parse_bool(value: Any, *, field: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ModelAPIError(
        422,
        "VALIDATION_ERROR",
        f"{field} must be a boolean.",
        details=[{"field": field, "issue": "boolean_required", "received": _safe_received(value)}],
    )


def parse_float(
    value: Any,
    *,
    field: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelAPIError(
            422,
            "VALIDATION_ERROR",
            f"{field} must be numeric.",
            details=[{"field": field, "issue": "number_required", "received": _safe_received(value)}],
        ) from exc
    if (minimum is not None and parsed < minimum) or (maximum is not None and parsed > maximum):
        raise ModelAPIError(
            422,
            "VALIDATION_ERROR",
            f"{field} is outside the allowed range.",
            details=[{"field": field, "minimum": minimum, "maximum": maximum, "received": parsed}],
        )
    return parsed


def parse_json_field(value: Any, *, field: str, default: Any = None) -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ModelAPIError(
            422,
            "VALIDATION_ERROR",
            f"{field} must contain valid JSON.",
            details=[{"field": field, "issue": "invalid_json"}],
        ) from exc


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
