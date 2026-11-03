"""Metadata-only audit events. Never accept document bodies or credential values."""
from __future__ import annotations

import functools
import inspect
import ipaddress
import json
import logging
import os
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from urllib.parse import urlsplit

audit_context = ContextVar("audit_context", default=None)
logger = logging.getLogger("model_audit")
logger.addHandler(logging.NullHandler())
logger.propagate = False


def configure_audit():
    path = os.getenv("MODEL_AUDIT_LOG_FILE")
    if not path or any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        return
    handler = RotatingFileHandler(path, maxBytes=20 * 1024 * 1024,
                                  backupCount=10, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def emit(event, **fields):
    # Call sites pass explicit metadata, never arbitrary headers/fields/exceptions.
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event,
              **(audit_context.get() or {}), **fields}
    record = {k: v[:1024] if isinstance(v, str) else v for k, v in record.items()}
    for key in ("asgi_client_ip", "rate_limit_client_key"):
        if record.get(key) is not None:
            try:
                record[key] = str(ipaddress.ip_address(record[key]))
            except ValueError:
                record[key] = "<invalid-or-unavailable>"
    logger.info(json.dumps(record, ensure_ascii=True, separators=(",", ":")))


def authentication(request, result, *, credential_slot=None, credential_type=None):
    context = audit_context.get()
    metadata = {"auth_result": result, "credential_slot": credential_slot,
                "credential_type": credential_type}
    if context is not None:
        context.update(metadata)
    emit("auth_" + result, **metadata)


def audit_upstream(fn):
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        parsed = urlsplit(bound["base_url"])
        fields = {"upstream_host": parsed.hostname,
                  "upstream_endpoint": bound.get("endpoint", "/paddle/api/v1/readiness").split("?", 1)[0],
                  "request_id": bound["request_id"],
                  "image_count": len(bound.get("image_paths", []))}
        emit("upstream_started", **fields)
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            emit("upstream_failed", **fields, duration_ms=round((time.perf_counter()-start)*1000, 2),
                 error_code=getattr(exc, "code", type(exc).__name__))
            raise
        emit("upstream_completed", **fields, duration_ms=round((time.perf_counter()-start)*1000, 2))
        return result
    return wrapper
