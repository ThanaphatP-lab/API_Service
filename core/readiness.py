from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any
from fastapi import FastAPI, Request
from shared.contracts import API_VERSION, ModelAPIError, request_id, success_response

logger = logging.getLogger('model_api')


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

