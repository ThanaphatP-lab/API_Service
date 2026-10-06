from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from shared.contracts import API_VERSION, ModelAPIError, request_id, success_response
from shared.rate_limit import InMemoryRateLimiter, client_key, rate_limit_headers
from core.auth import _AUTH_EXEMPT_PATHS, _configured_tokens, _request_token, _security_headers, token_matches
from core.errors import _error_body, _safe_received, _public_error_details
from core.settings import _is_production, _env_flag, runtime_settings
from core.limits import check_request_size
from core.telemetry import RequestTelemetry, configure_access_logging, request_timings

logger = logging.getLogger('model_api')


def create_app(
    title: str,
    model_name: str,
    *,
    service_name: str | None = None,
    auth_token_env: str = "INTERNAL_API_TOKEN",
    required_auth_token_envs: tuple[str, ...] = (),
    public_api: bool = False,
    max_concurrent_requests: int | None = None,
    lifespan: Any = None,
) -> FastAPI:
    service = service_name or title.lower().replace(" ", "-")
    configure_access_logging()
    docs_enabled = not _is_production() and _env_flag("API_DOCS_ENABLED", "true")
    app = FastAPI(
        title=title,
        version="1.0.0",
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        lifespan=lifespan,
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
    origins = runtime_settings.cors_origins
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
    max_concurrent = max_concurrent_requests if max_concurrent_requests is not None else runtime_settings.max_concurrent_requests(public_api=public_api)
    if max_concurrent < 1:
        raise ValueError("max_concurrent_requests must be positive")
    queue_timeout = runtime_settings.queue_timeout
    inference_slots = asyncio.Semaphore(max_concurrent)

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[..., Any]):
        supplied = request.headers.get("x-request-id", "").strip()
        valid_supplied = bool(
            supplied
            and len(supplied) <= 128
            and supplied.replace("-", "").replace("_", "").isalnum()
        )
        request.state.request_id = getattr(request.state, "request_id", None) or (supplied if valid_supplied else f"req_{uuid.uuid4().hex}")
        request.state.started_at = time.perf_counter()
        documentation_path = docs_enabled and request.url.path in {"/docs", "/redoc", "/openapi.json"}

        legacy_aliases = {"/predict", "/v1/textdetection", "/v1/textrecognition"}
        if request.url.path in legacy_aliases and not _env_flag(
            "LEGACY_ENDPOINTS_ENABLED", "false" if _is_production() else "true"
        ):
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
            if not token_matches(provided_token, expected_tokens):
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
        try:
            check_request_size(content_length)
        except ModelAPIError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content=_error_body(request, code=exc.code, message=exc.message, details=exc.details),
                headers=_security_headers({"X-Request-ID": request.state.request_id}),
            )

        acquired = False
        queue_started = time.perf_counter()
        if request.method == "POST":
            try:
                await asyncio.wait_for(inference_slots.acquire(), timeout=max(0.1, queue_timeout))
                acquired = True
                timings = request_timings.get()
                if timings is not None:
                    timings["admission"] = (time.perf_counter()-queue_started)*1000
                logging.getLogger("uvicorn.error").debug(
                    "admission_complete request_id=%s service=%s queue_wait_ms=%.2f",
                    request.state.request_id, service, (time.perf_counter()-queue_started)*1000,
                )
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
        logger.debug(
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
        if runtime_settings.debug_errors:
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
                "device": runtime_settings.model_device,
            },
            service=service,
            model=model_name,
        )

    @app.get("/health", include_in_schema=False)
    def legacy_health(request: Request) -> dict[str, Any]:
        return health(request)

    app.add_middleware(RequestTelemetry, service=service)
    return app
