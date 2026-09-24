from __future__ import annotations

import os
import hmac
from fastapi import Request
from shared.contracts import API_VERSION
from core.settings import _is_production


_AUTH_EXEMPT_PATHS = {"/health", f"/api/{API_VERSION}/health"}


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


def token_matches(provided: str, expected: tuple[str, ...]) -> bool:
    return bool(provided) and any(hmac.compare_digest(provided, token) for token in expected)
