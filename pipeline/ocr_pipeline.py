from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shared.upstream import chunked_paths, post_images


def _value(result: dict[str, Any], key: str) -> Any:
    return result.get(key, result.get("res", {}).get(key))


def _crop_quad(image: np.ndarray, points: Any) -> np.ndarray | None:
    quad = np.asarray(points, dtype=np.float32).copy()
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return None

    area = 0.0
    for index in range(-1, 3):
        area += -0.5 * (quad[index + 1][1] + quad[index][1]) * (quad[index + 1][0] - quad[index][0])
    if area < 0:
        quad[[1, 3]] = quad[[3, 1]]

    width = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3])))
    height = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))
    if width <= 0 or height <= 0:
        return None

    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    crop = cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(quad, target),
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.size == 0:
        return None
    crop_height, crop_width = crop.shape[:2]
    if crop_height / float(max(crop_width, 1)) >= 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


def _text_from_recognition(result: dict[str, Any]) -> tuple[str | None, float | None]:
    texts = _value(result, "rec_texts")
    scores = _value(result, "rec_scores")
    # The standalone module returns singular keys, whereas the OCR pipeline
    # returns plural keys. Accept both result shapes.
    if texts is None:
        texts = _value(result, "rec_text")
    if scores is None:
        scores = _value(result, "rec_score")
    text = texts[0] if isinstance(texts, list) and texts else texts
    score = scores[0] if isinstance(scores, list) and scores else scores
    return text, score


def legacy_ocr_contract(
    lines: list[dict[str, Any]],
    *,
    engine: str,
    model: str = "th_PP-OCRv5_mobile_rec",
    detection: list[dict[str, Any]] | None = None,
    upstreams: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return the OCR shape consumed by the original Backend.

    The v1 model APIs keep ``lines`` as their canonical geometry payload, while
    these compatibility fields let old application flows consume the same
    response without fabricating another Backend-side OCR pass.
    """
    normalized_lines: list[dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = str(line.get("text") or "")
        try:
            confidence = float(line.get("rec_score") if line.get("rec_score") is not None else line.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        normalized_lines.append({**line, "text": text, "rec_score": confidence, "confidence": confidence})

    non_empty = [line for line in normalized_lines if line["text"].strip()]
    confidence = (
        sum(float(line["confidence"]) for line in non_empty) / len(non_empty)
        if non_empty
        else 0.0
    )
    result: dict[str, Any] = {
        "text": "\n".join(line["text"] for line in non_empty),
        "confidence": float(confidence),
        "segments": normalized_lines,
        "raw_segments": normalized_lines,
        "lines": normalized_lines,
        "preprocessing": "polygon_perspective_crop_then_paddle_recognition",
        "engine": engine,
        "model": model,
        "predictions": normalized_lines,
    }
    if detection is not None:
        result["detection"] = detection
    if upstreams is not None:
        result["upstreams"] = upstreams
    return result


def direct_recognition_contract(
    prediction: dict[str, Any] | None,
    *,
    model: str = "th_PP-OCRv5_mobile_rec",
) -> dict[str, Any]:
    """Normalize one recognition-only result to the legacy ROI OCR contract."""
    prediction = prediction if isinstance(prediction, dict) else {}
    text, score = _text_from_recognition(prediction)
    try:
        confidence = float(score or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    segment = {"text": str(text or ""), "confidence": confidence, "rec_score": confidence}
    return {
        "text": segment["text"],
        "confidence": confidence,
        "segments": [segment] if segment["text"] else [],
        "raw_segments": [prediction] if prediction else [],
        "predictions": [prediction] if prediction else [],
        "preprocessing": "recognition_only",
        "engine": "paddle_thai_ocr",
        "model": model,
    }


def predict_custom_ocr(image_path: str) -> dict[str, Any]:
    """Detect text first, perspective-crop each polygon, then recognize it.

    This intentionally keeps detection and recognition independent so each model
    can be replaced with a local fine-tuned export without changing the API.
    """
    # Local fallback only. The production pipeline uses HTTP and therefore
    # does not import Paddle in its lightweight orchestration environment.
    from models.det_model import predict_detection
    from models.rec_model import predict_recognition

    detections = predict_detection(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("OpenCV could not decode the uploaded image")

    lines: list[dict[str, Any]] = []
    for detection in detections:
        polygons = _value(detection, "dt_polys") or []
        scores = _value(detection, "dt_scores") or []
        for index, polygon in enumerate(polygons):
            crop = _crop_quad(image, polygon)
            if crop is None or crop.size == 0:
                continue
            # TextRecognition accepts a NumPy BGR image, avoiding temporary crops.
            recognized = predict_recognition(crop)
            text, rec_score = _text_from_recognition(recognized[0]) if recognized else (None, None)
            lines.append(
                {
                    "polygon": polygon,
                    "det_score": scores[index] if index < len(scores) else None,
                    "text": text,
                    "rec_score": rec_score,
                }
            )
    return legacy_ocr_contract(
        lines,
        engine="custom-det-rec",
        detection=detections,
    )


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
) -> dict[str, Any]:
    """Compose independently scalable detector and recognizer HTTP services."""
    detection_fields = {
        "model": det_model if det_model is not None else model,
    }
    recognition_fields = {
        "model": rec_model if rec_model is not None else model,
    }
    if version is not None:
        detection_fields["version"] = version
        recognition_fields["version"] = version

    detection_data = post_images(
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
                recognition_data = post_images(
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
