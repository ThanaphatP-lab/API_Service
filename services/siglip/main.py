from __future__ import annotations

import os
from typing import Any

import torch
from fastapi import Request
from PIL import Image
from transformers import SiglipModel, SiglipProcessor

from shared.api import IMAGE_REQUEST_OPENAPI, add_readiness_route, create_app, parse_image_request, singleflight_lru_cache
from shared.contracts import ModelAPIError, success_response
from shared.inference_adapters import adapt_siglip
from shared.settings import model_dir
from shared.siglip_categories import siglip_prompts

MODEL_NAME = os.getenv("SIGLIP_MODEL_NAME", "google/siglip-so400m-patch14-384")
SERVICE_NAME = "siglip-classification-model"
app = create_app("SigLIP Zero-shot Classification API", MODEL_NAME, service_name=SERVICE_NAME)


def _torch_device() -> str:
    configured = os.getenv("MODEL_DEVICE", "gpu:0")
    if configured.startswith("gpu") or configured.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("MODEL_DEVICE requests GPU but CUDA is not available")
        index = configured.split(":", 1)[1] if ":" in configured else "0"
        return f"cuda:{index}"
    return "cpu"


@singleflight_lru_cache(maxsize=1)
def components() -> tuple[Any, Any, str]:
    source = model_dir("SIGLIP_MODEL_DIR") or MODEL_NAME
    target = _torch_device()
    processor = SiglipProcessor.from_pretrained(source)
    model = SiglipModel.from_pretrained(source).eval().to(target)
    return processor, model, target


@app.post("/api/v1/image-classifications", tags=["Model inference"], openapi_extra=IMAGE_REQUEST_OPENAPI)
@app.post("/predict", include_in_schema=False)
async def predict(request: Request) -> dict[str, Any]:
    image = await parse_image_request(request)
    try:
        # Notebook (4) accepts category objects and evaluates only enabled
        # prompts. ``labels`` remains a compatibility alias for the existing
        # image-verification pipeline.
        field_name = "categories" if image.fields.get("categories") is not None else "labels"
        category_value = image.fields.get(field_name)
        try:
            candidates = siglip_prompts(category_value)
        except ValueError as exc:
            raise ModelAPIError(
                422,
                "VALIDATION_ERROR",
                "SigLIP categories are invalid.",
                details=[{"field": field_name, "issue": str(exc)[:200]}],
            ) from exc
        if not candidates:
            raise ModelAPIError(
                422,
                "LABELS_REQUIRED",
                "Provide at least one enabled SigLIP category or label.",
                details=[{"field": field_name, "issue": "required"}],
            )

        processor, model, target = components()
        with Image.open(image.path) as source:
            image_value = source.convert("RGB")
        inputs = processor(text=candidates, images=image_value, padding="max_length", return_tensors="pt")
        inputs = {key: value.to(target) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits_per_image[0]
            raw_logits = logits.detach().cpu().tolist()
            probabilities = torch.sigmoid(logits).detach().cpu().tolist()
        predictions = [
            {"label": label, "score": score, "logit": logit}
            for label, score, logit in sorted(
                zip(candidates, probabilities, raw_logits),
                key=lambda item: item[2],
                reverse=True,
            )
        ]
        payload = adapt_siglip(logits=raw_logits, categories=candidates, device=target)
        payload["predictions"] = predictions
        return success_response(
            request,
            payload,
            service=SERVICE_NAME,
            model=MODEL_NAME,
        )
    except Exception as exc:
        if isinstance(exc, ModelAPIError):
            raise
        raise ModelAPIError(
            500,
            "MODEL_INFERENCE_FAILED",
            "SigLIP could not complete inference.",
            details=[{"reason": str(exc)[:500]}],
        ) from exc
    finally:
        image.cleanup()


add_readiness_route(app, components)
