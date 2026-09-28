from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
import cv2

from clients.model_service_client import HTTPModelClient, ModelClient
from shared.upstream import chunked_paths
from pipelines.ocr.polygon_crop import _crop_quad
from pipelines.ocr.contracts import _value, _text_from_recognition, legacy_ocr_contract




def predict_remote_ocr(
    image_path: Path,
    *,
    detector_url: str,
    recognizer_url: str,
    request_id: str,
    version: str | None = None,
    model: str = "baseline",
    det_model: str | None = None,
    rec_model: str | None = None,
    recognition_batch_size: int = 64,
    client: ModelClient | None = None,
) -> dict[str, Any]:
    """Compose independently scalable detector and recognizer HTTP services."""
    client = client if client is not None else HTTPModelClient()
    detection_fields = {
        "model": det_model if det_model is not None else model,
    }
    recognition_fields = {
        "model": rec_model if rec_model is not None else model,
    }
    if version is not None:
        detection_fields["version"] = version
        recognition_fields["version"] = version

    detection_data = client.infer(
        detector_url,
        "/api/v1/text-detections",
        [image_path],
        fields=detection_fields,
        request_id=request_id,
    )
    detections = detection_data.get("predictions") or []
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("OpenCV could not decode the uploaded image")

    crop_paths: list[Path] = []
    crop_metadata: list[dict[str, Any]] = []
    try:
        for detection in detections:
            polygons = _value(detection, "dt_polys") or []
            scores = _value(detection, "dt_scores") or []
            for index, polygon in enumerate(polygons):
                crop = _crop_quad(image, polygon)
                if crop is None or crop.size == 0:
                    continue
                handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                handle.close()
                crop_path = Path(handle.name)
                if not cv2.imwrite(str(crop_path), crop):
                    crop_path.unlink(missing_ok=True)
                    continue
                crop_paths.append(crop_path)
                crop_metadata.append(
                    {
                        "polygon": polygon,
                        "det_score": scores[index] if index < len(scores) else None,
                    }
                )

        recognition_predictions: list[dict[str, Any]] = []
        recognition_selection: dict[str, Any] | None = None
        if crop_paths:
            for batch in chunked_paths(crop_paths, recognition_batch_size):
                recognition_data = client.infer(
                    recognizer_url,
                    "/api/v1/text-recognition-batches",
                    batch,
                    fields=recognition_fields,
                    request_id=request_id,
                    multiple=True,
                )
                batch_predictions = recognition_data.get("predictions") or []
                if isinstance(batch_predictions, list):
                    recognition_predictions.extend(batch_predictions)
                current_selection = recognition_data.get("model_selection")
                if isinstance(current_selection, dict):
                    recognition_selection = current_selection

        lines = []
        for index, metadata in enumerate(crop_metadata):
            result = recognition_predictions[index] if index < len(recognition_predictions) else {}
            text, rec_score = _text_from_recognition(result)
            lines.append({**metadata, "text": text, "rec_score": rec_score})
        result = legacy_ocr_contract(
            lines,
            engine="remote-det-rec-v1",
            model=(
                str(recognition_selection.get("model_name"))
                if recognition_selection and recognition_selection.get("model_name")
                else recognition_fields["model"]
            ),
            detection=detections,
            upstreams={
                "detector": detector_url,
                "recognizer": recognizer_url,
            },
        )
        result["model_selection"] = {
            "version": version,
            "profile": model,
            "detection": detection_data.get("model_selection"),
            "recognition": recognition_selection,
        }
        return result
    finally:
        for path in crop_paths:
            path.unlink(missing_ok=True)
