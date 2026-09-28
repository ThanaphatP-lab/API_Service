from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Request

from inference.text_recognition import get_model, infer, infer_batch
from core.request_parsing import BATCH_IMAGE_REQUEST_OPENAPI, IMAGE_REQUEST_OPENAPI, parse_image_request
from core.fair_queue import FairInferenceQueue
from core.settings import RecognitionQueueSettings
from core.app_factory import create_app
from core.errors import run_image_inference
from shared.contracts import ModelAPIError, request_id, success_response
from shared.model_variants import resolve_model_variant

MODEL_NAME = os.getenv("REC_MODEL_NAME", "th_PP-OCRv5_mobile_rec")
SERVICE_NAME = "leaf-text-recognition"
queue_settings = RecognitionQueueSettings()


@asynccontextmanager
async def lifespan(app):
    queue = FairInferenceQueue(
        quantum=queue_settings.quantum, max_jobs=queue_settings.max_jobs,
        max_items=queue_settings.max_items, max_bytes=queue_settings.max_bytes,
        timeout=queue_settings.timeout,
    )
    app.state.rec_queue = queue
    try:
        yield
    finally:
        await queue.aclose()


app = create_app("Thai Text Recognition API", MODEL_NAME, service_name=SERVICE_NAME,
                 max_concurrent_requests=queue_settings.http_requests, lifespan=lifespan)
logger = logging.getLogger("uvicorn.error")


async def queued_inference(request, images, selected, *, multiple):
    """Takes ownership of images even when admission fails or caller is cancelled."""
    try:
        queue = request.app.state.rec_queue
        paths = list(images.paths)
        cost_bytes = sum(path.stat().st_size for path in paths)
    except Exception:
        images.cleanup()
        raise

    def run(paths):
        return run_image_inference(
            lambda _: infer_batch([str(path) for path in paths], selected) if multiple
            else infer(str(paths[0]), selected), paths[0],
        )

    chunks = await queue.submit(paths, run, cleanup=images.cleanup, cost_bytes=cost_bytes,
                                label=request_id(request))
    if not multiple or len(chunks) == 1:
        return chunks[0]
    results = [item for chunk in chunks for item in chunk["results"]]
    return {**chunks[0], "result": {"results": results}, "results": results,
            "predictions": results, "count": len(results),
            "raw_output": [item for chunk in chunks for item in chunk.get("raw_output", [])]}


@app.post("/api/v1/text-recognitions", tags=["Model inference"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/v1/textrecognition", include_in_schema=False)
@app.post("/v1/textdetection", include_in_schema=False)
@app.post("/predict", include_in_schema=False)
async def predict(
    request: Request,
    version: str | None = None,
    model: str = "baseline",
) -> dict:
    image = await parse_image_request(request)
    owned_by_queue = False
    try:
        selected = resolve_model_variant(
            "recognition",
            version if version is not None else image.fields.get("version"),
            request.query_params.get("model", image.fields.get("model", model)),
        )
        logger.info(
            "Recognition request selection endpoint=single version=%s variant=%s model_dir=%s local_weights=%s",
            selected.version,
            selected.variant,
            str(selected.model_dir) if selected.model_dir is not None else "<official-model-cache>",
            selected.model_dir is not None,
        )
        owned_by_queue = True
        payload = await queued_inference(request, image, selected, multiple=False)
        response_selection = payload.get("model_selection", selected.public_dict())
        logger.info(
            "Recognition response endpoint=single version=%s variant=%s response_model=%s model_name=%s model_dir=%s local_weights=%s result_count=%s",
            selected.version,
            selected.variant,
            selected.response_model,
            response_selection.get("model_name"),
            str(selected.model_dir) if selected.model_dir is not None else "<official-model-cache>",
            selected.model_dir is not None,
            len(payload.get("predictions") or []),
        )
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=selected.response_model,
            model_version=selected.version,
        )
    finally:
        if not owned_by_queue:
            image.cleanup()


@app.post("/api/v1/text-recognition-batches", tags=["Model inference"], openapi_extra=BATCH_IMAGE_REQUEST_OPENAPI)
async def recognize_batch(
    request: Request,
    version: str | None = None,
    model: str = "baseline",
) -> dict:
    images = await parse_image_request(request, multiple=True)
    owned_by_queue = False
    try:
        selected = resolve_model_variant(
            "recognition",
            version if version is not None else images.fields.get("version"),
            request.query_params.get("model", images.fields.get("model", model)),
        )
        logger.info(
            "Recognition request selection endpoint=batch version=%s variant=%s model_dir=%s local_weights=%s image_count=%s",
            selected.version,
            selected.variant,
            str(selected.model_dir) if selected.model_dir is not None else "<official-model-cache>",
            selected.model_dir is not None,
            len(images.paths),
        )
        owned_by_queue = True
        payload = await queued_inference(request, images, selected, multiple=True)
        response_selection = payload.get("model_selection", selected.public_dict())
        logger.info(
            "Recognition response endpoint=batch version=%s variant=%s response_model=%s model_name=%s model_dir=%s local_weights=%s image_count=%s result_count=%s",
            selected.version,
            selected.variant,
            selected.response_model,
            response_selection.get("model_name"),
            str(selected.model_dir) if selected.model_dir is not None else "<official-model-cache>",
            selected.model_dir is not None,
            len(images.paths),
            len(payload.get("results") or []),
        )
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=selected.response_model,
            model_version=selected.version,
        )
    finally:
        if not owned_by_queue:
            images.cleanup()


@app.get("/api/v1/readiness", tags=["Operations"])
async def readiness(request: Request):
    def load(_):
        try:
            get_model(resolve_model_variant("recognition", None, "baseline"))
        except Exception as exc:
            raise ModelAPIError(503, "MODEL_NOT_READY", "The model could not be loaded.") from exc
    # Keep loading/readiness on the same worker as inference, not another GPU thread.
    await request.app.state.rec_queue.submit([None], load, label=request_id(request))
    return success_response(request, {"status": "ready"}, service=SERVICE_NAME, model=MODEL_NAME)
