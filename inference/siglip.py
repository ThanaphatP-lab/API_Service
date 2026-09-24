from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import SiglipModel, SiglipProcessor

from core.cache import singleflight_lru_cache
from shared.inference_adapters import adapt_siglip
from shared.settings import model_dir


@dataclass(frozen=True)
class SiglipSelection:
    model_name: str
    model_dir: str | None
    device: str


@dataclass(frozen=True)
class SiglipRuntime:
    processor: Any
    model: Any
    device: str


def _torch_device() -> str:
    configured = os.getenv("MODEL_DEVICE", "gpu:0")
    if configured.startswith(("gpu", "cuda")):
        if not torch.cuda.is_available():
            raise RuntimeError("MODEL_DEVICE requests GPU but CUDA is not available")
        index = configured.split(":", 1)[1] if ":" in configured else "0"
        return f"cuda:{index}"
    return "cpu"


def selection_from_settings() -> SiglipSelection:
    return SiglipSelection(
        model_name=os.getenv(
            "SIGLIP_MODEL_NAME",
            "google/siglip-so400m-patch14-384",
        ),
        model_dir=model_dir("SIGLIP_MODEL_DIR"),
        device=_torch_device(),
    )


@singleflight_lru_cache(maxsize=1)
def _load_model(selection: SiglipSelection) -> SiglipRuntime:
    source = selection.model_dir or selection.model_name
    processor = SiglipProcessor.from_pretrained(source)
    model = SiglipModel.from_pretrained(source).eval().to(selection.device)
    return SiglipRuntime(processor=processor, model=model, device=selection.device)


def get_model(selection: SiglipSelection | None = None) -> SiglipRuntime:
    return _load_model(selection or selection_from_settings())


def infer(
    image_path: str,
    candidates: list[str],
    selection: SiglipSelection | None = None,
) -> dict[str, Any]:
    selected = selection or selection_from_settings()
    runtime = get_model(selected)
    with Image.open(image_path) as source:
        image_value = source.convert("RGB")
    inputs = runtime.processor(
        text=candidates,
        images=image_value,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {key: value.to(runtime.device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = runtime.model(**inputs).logits_per_image[0]
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
    payload = adapt_siglip(
        logits=raw_logits,
        categories=candidates,
        device=runtime.device,
    )
    payload["predictions"] = predictions
    return payload
