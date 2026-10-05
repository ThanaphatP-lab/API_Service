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
            logger.info("stage_complete request_id=%s stage=%s duration_ms=%.2f outcome=%s",
                        current_request.get(), name, (time.perf_counter() - start) * 1000, outcome)
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
        start, status, received, sent = time.perf_counter(), 500, 0, 0
        complete = False
        logger.info("request_received request_id=%s service=%s method=%s path=%s",
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
            logger.info("request_complete request_id=%s service=%s status=%s duration_ms=%.2f received_bytes=%s sent_bytes=%s response_complete=%s resources=%s",
                        rid, self.service, status, (time.perf_counter()-start)*1000,
                        received, sent, complete, resources())
            current_request.reset(token)
