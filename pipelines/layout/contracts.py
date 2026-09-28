"""Legacy DET response formatting without an additional HTTP pipeline hop."""
from typing import Any
from pipelines.layout.geometry import _text_regions


def format_legacy_detection(payload, paths, *, multiple=False):
    # Runs inside the Paddle leaf runtime, not the lightweight API Gateway.
    import cv2
    from shared.contracts import ModelAPIError

    predictions = payload.get("predictions") or []
    results = []
    for index, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            if not multiple:
                raise ModelAPIError(422, "INVALID_IMAGE", "OpenCV could not decode the image.")
            results.append({"engine": "paddleocr", "model": "PP-OCRv5_server_det",
                            "image_width": 0, "image_height": 0, "regions": []})
            continue
        height, width = image.shape[:2]
        selected = predictions
        if multiple:
            selected = [predictions[index] if index < len(predictions) and isinstance(predictions[index], dict) else {}]
        results.append(legacy_detection_contract({"predictions": selected}, width, height))
    return {"results": results, "count": len(results)} if multiple else results[0]


def legacy_detection_contract(detection_data: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    predictions = detection_data.get("predictions") or []
    regions = _text_regions(
        predictions,
        width,
        height,
        expand=False,
        padding=(0, 0, 0, 0),
    )
    legacy_regions = []
    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        legacy_regions.append(
            {
                "text": "",
                "confidence": region["score"],
                "bbox": {
                    "x": x1,
                    "y": y1,
                    "width": round(x2 - x1, 2),
                    "height": round(y2 - y1, 2),
                    "x_ratio": region["bbox_ratio"][0],
                    "y_ratio": region["bbox_ratio"][1],
                    "width_ratio": round((x2 - x1) / width, 6),
                    "height_ratio": round((y2 - y1) / height, 6),
                },
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "type": "text",
                "data_type": "text",
                "label": "text",
                "source": "text-detection",
            }
        )
    return {
        "engine": "paddleocr",
        "model": "PP-OCRv5_server_det",
        "image_width": width,
        "image_height": height,
        "regions": legacy_regions,
        "raw_predictions": predictions,
    }
