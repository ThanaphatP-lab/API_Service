"""Low-overhead timings; never log image contents, credentials, or query strings."""
from __future__ import annotations

import functools
import inspect
import logging
import os
import time
from contextvars import ContextVar

logger = logging.getLogger("uvicorn.error")
current_request = ContextVar("telemetry_request", default="-")
request_timings = ContextVar("telemetry_timings", default=None)


class ManagedAccessFilter(logging.Filter):
    """Managed requests already have a completion summary; leave other apps alone."""
    def filter(self, record):
        return (current_request.get() == "-" or
                os.getenv("API_ACCESS_LOG", "false").lower() in {"1", "true", "yes", "on"})


def configure_access_logging():
    # Application logs use the same handler/level as Uvicorn, without enabling
    # noisy DEBUG output from all third-party libraries via the root logger.
    app_logger = logging.getLogger("model_api")
    server_logger = logging.getLogger("uvicorn.error")
    configured = os.getenv("APP_LOG_LEVEL", "").lower()
    app_logger.setLevel(logging.DEBUG if configured == "debug" else
                        logging.INFO if configured == "info" else server_logger.getEffectiveLevel())
    handler_owner = server_logger
    while not handler_owner.handlers and handler_owner.parent is not None:
        handler_owner = handler_owner.parent
    if handler_owner.handlers:
        app_logger.handlers = list(handler_owner.handlers)
        app_logger.propagate = False
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, ManagedAccessFilter) for f in access.filters):
        access.addFilter(ManagedAccessFilter())


def resources():
    """Process snapshots, not per-request allocation or GPU measurements."""
    result = {"pid": os.getpid(), "process_cpu_seconds": round(time.process_time(), 3)}
    try:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_peak_rss_mb"] = round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 2)
    except (ImportError, OSError, ValueError):
        pass
    return result


def timed_stage(name):
    def decorate(fn):
        def report(start, outcome):
            elapsed = (time.perf_counter() - start) * 1000
            timings = request_timings.get()
            if timings is not None:
                timings[name] = timings.get(name, 0.0) + elapsed
            logger.debug("stage_complete request_id=%s stage=%s duration_ms=%.2f outcome=%s",
                         current_request.get(), name, elapsed, outcome)
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                start, outcome = time.perf_counter(), "error"
                try:
                    result = await fn(*args, **kwargs)
                    outcome = "ok"
                    return result
                finally:
                    report(start, outcome)
            return async_wrapper
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start, outcome = time.perf_counter(), "error"
            try:
                result = fn(*args, **kwargs)
                outcome = "ok"
                return result
            finally:
                report(start, outcome)
        return wrapper
    return decorate


class RequestTelemetry:
    def __init__(self, app, service):
        self.app, self.service = app, service

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        import uuid
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"x-request-id", b"").decode("latin1").strip()
        rid = supplied if supplied and len(supplied) <= 128 and supplied.replace("-", "").replace("_", "").isalnum() else "req_" + uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        token = current_request.set(rid)
        timings = {}
        timings_token = request_timings.set(timings)
        start, status, received, sent = time.perf_counter(), 500, 0, 0
        complete = False
        logger.debug("request_received request_id=%s service=%s method=%s path=%s",
                    rid, self.service, scope["method"], scope["path"])
        async def measured_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            return message
        async def measured_send(message):
            nonlocal status, sent, complete
            if message["type"] == "http.response.start":
                status = message["status"]
            if message["type"] == "http.response.body":
                sent += len(message.get("body", b""))
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                complete = True
        try:
            await self.app(scope, measured_receive, measured_send)
        finally:
            health = scope["path"] in {"/health", "/api/v1/health", "/api/v1/readiness"}
            log = logger.warning if status >= 400 or not complete else logger.debug if health else logger.info
            stage_names = {"receive_decode_verify_images": "receive", "upstream_http_roundtrip": "upstream",
                           "inference_including_load_and_adaptation": "infer", "admission": "queue"}
            stages = " ".join(f"{stage_names.get(k, k)}={v:.0f}ms" for k, v in timings.items())
            log("request_complete service=%s %s %s status=%s total=%.0fms %s in=%.1fKiB out=%.1fKiB request_id=%s complete=%s",
                self.service, scope["method"], scope["path"], status,
                (time.perf_counter()-start)*1000, stages, received/1024, sent/1024, rid, complete)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("request_resources request_id=%s resources=%s received_bytes=%s sent_bytes=%s",
                             rid, resources(), received, sent)
            request_timings.reset(timings_token)
            current_request.reset(token)
