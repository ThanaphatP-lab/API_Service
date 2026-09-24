from __future__ import annotations

from pathlib import Path
from typing import Any

from clients.model_service_client import ModelClient
from pipelines.verification.scoring import verify_classification_targets


def verify_image(
    image_path: Path, *, targets: list[str], categories: list[dict[str, Any]],
    classifier_url: str, request_id: str, client: ModelClient,
) -> dict[str, Any]:
    classification = client.infer(
        classifier_url, "/api/v1/image-classifications", [image_path],
        fields={"labels": [item["prompt"] for item in categories]},
        request_id=request_id,
    )
    results = verify_classification_targets(
        target_values=targets, categories=categories, classification=classification,
    )
    selected = next((item for item in results if item.get("passed")), None)
    if selected is None:
        selected = max(results, key=lambda item: float(item.get("evidence_score") or 0.0))
    return {**selected, "verifications": results, "requested_categories": targets}
