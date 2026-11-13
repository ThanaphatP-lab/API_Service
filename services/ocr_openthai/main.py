"""OpenThai OCR using shared API infrastructure."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from fastapi import Request
from fastapi.responses import Response
from core.app_factory import create_app as create_model_app
from core.request_parsing import IMAGE_REQUEST_OPENAPI, parse_image_request
from inference.openthai.engine import Inference
from inference.openthai.settings import MODEL_REVISION, Settings
from shared.contracts import ModelAPIError, success_response

SERVICE_NAME = "leaf-openthai-ocr"
MODEL_NAME = "iapp/OpenThai2.0-OCR-Text"
ENDPOINT = "/paddle/api/v1/openthai-text"


def create_app(settings=None, loader=None):
    settings = settings or Settings()
    inference = Inference(settings, loader)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await asyncio.get_running_loop().run_in_executor(inference.executor, inference.initialize)
        except Exception:
            logging.getLogger(__name__).exception("OpenThai initialization failed")
        try:
            yield
        finally:
            inference.executor.shutdown(wait=True, cancel_futures=True)

    app = create_model_app("OpenThai OCR", MODEL_NAME, service_name=SERVICE_NAME,
                           auth_token_env="INTERNAL_API_TOKEN",
                           max_concurrent_requests=1, lifespan=lifespan)
    app.state.inference = inference

    def ensure_ready():
        if inference.reader is None:
            raise ModelAPIError(503, "MODEL_NOT_READY", "The OpenThai model is unavailable.")

    @app.get("/paddle/api/v1/readiness", tags=["Operations"])
    async def readiness(request: Request):
        ensure_ready()
        return success_response(request, {"status": "ready"}, service=SERVICE_NAME,
                                model=MODEL_NAME, model_version=MODEL_REVISION)

    @app.post(ENDPOINT, tags=["Model inference"], openapi_extra=IMAGE_REQUEST_OPENAPI)
    async def predict(request: Request):
        ensure_ready()
        image = await parse_image_request(request)
        try:
            # Retain a concurrent Future so HTTP cancellation cannot release
            # the native inference slot or enqueue additional model work.
            if inference.pending is not None and not inference.pending.done():
                raise ModelAPIError(503, "SERVICE_BUSY", "OpenThai is still processing a previous request.")
            if image.path.stat().st_size > settings.max_upload_bytes:
                raise ModelAPIError(413, "PAYLOAD_TOO_LARGE", "Image exceeds the OpenThai upload limit.")
            inference.pending = inference.executor.submit(inference.read, image.path.read_bytes())
            future = asyncio.wrap_future(inference.pending)
            future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            try:
                data = await asyncio.wait_for(asyncio.shield(future), settings.request_timeout_seconds)
            except TimeoutError as exc:
                raise ModelAPIError(504, "REQUEST_TIMEOUT", "OpenThai processing deadline exceeded.") from exc
            except ModelAPIError:
                raise
            except Exception as exc:
                raise ModelAPIError(500, "MODEL_INFERENCE_FAILED", "OpenThai could not complete inference.") from exc
            payload = success_response(request, data, service=SERVICE_NAME,
                                       model=MODEL_NAME, model_version=MODEL_REVISION)
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if len(body) > settings.max_response_bytes:
                raise ModelAPIError(503, "OUTPUT_TOO_LARGE", "Model output exceeds the response limit.")
            return Response(body, media_type="application/json")
        finally:
            image.cleanup()

    return app


app = create_app()
