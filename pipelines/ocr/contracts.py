from __future__ import annotations

from typing import Any


def _value(result: dict[str, Any], key: str) -> Any:
    return result.get(key, result.get("res", {}).get(key))


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
