from pathlib import Path
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from pipelines.ocr import orchestrator as ocr_pipeline
from services.ocr_pipeline_custom import main as custom_service
from shared.upstream import chunked_paths


def test_chunked_paths_preserves_order_and_remainder():
    paths = [Path(f"crop-{index}.png") for index in range(5)]

    assert chunked_paths(paths, 2) == [paths[:2], paths[2:4], paths[4:]]


def test_remote_ocr_chunks_recognition_and_forwards_model_selection(monkeypatch, tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    calls = []
    counter = {"value": 0}

    monkeypatch.setattr(
        ocr_pipeline.cv2,
        "imread",
        lambda _: np.ones((10, 10, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        ocr_pipeline,
        "_crop_quad",
        lambda image, polygon: np.ones((3, 5, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(ocr_pipeline.cv2, "imwrite", lambda path, image: Path(path).write_bytes(b"crop") > 0)

    def post_images(base_url, endpoint, image_paths, *, fields=None, request_id, multiple=False):
        calls.append(
            {
                "base_url": base_url,
                "endpoint": endpoint,
                "count": len(image_paths),
                "fields": fields,
                "multiple": multiple,
            }
        )
        if endpoint == "/api/v1/text-detections":
            polygons = [
                [[0, 0], [5, 0], [5, 3], [0, 3]]
                for _ in range(5)
            ]
            return {
                "predictions": [{"dt_polys": polygons, "dt_scores": [0.9] * 5}],
                "model_selection": {
                    "version": "v6",
                    "variant": "thai_det",
                    "model_name": "det-model",
                },
            }

        predictions = []
        for _ in image_paths:
            index = counter["value"]
            counter["value"] += 1
            predictions.append({"rec_text": f"line-{index}", "rec_score": 0.8})
        return {
            "predictions": predictions,
            "model_selection": {
                "version": "v6",
                "variant": "thai_rec",
                "model_name": "rec-model",
            },
        }

    class FakeClient:
        infer = staticmethod(post_images)

    result = ocr_pipeline.predict_remote_ocr(
        source,
        detector_url="http://det",
        recognizer_url="http://rec",
        request_id="req_contract",
        version="v6",
        model="baseline",
        det_model="thai_det",
        rec_model="thai_rec",
        recognition_batch_size=2,
        client=FakeClient(),
    )

    assert [call["count"] for call in calls] == [1, 2, 2, 1]
    assert calls[0]["fields"] == {"version": "v6", "model": "thai_det"}
    assert all(
        call["fields"] == {"version": "v6", "model": "thai_rec"}
        for call in calls[1:]
    )
    assert [line["text"] for line in result["lines"]] == [
        "line-0",
        "line-1",
        "line-2",
        "line-3",
        "line-4",
    ]
    assert result["model"] == "rec-model"
    assert result["model_selection"]["detection"]["variant"] == "thai_det"
    assert result["model_selection"]["recognition"]["variant"] == "thai_rec"


def test_custom_ocr_service_reads_model_selection_from_multipart(monkeypatch):
    captured = {}

    def remote_ocr(image_path, **kwargs):
        captured.update(kwargs)
        return {"text": "ok", "lines": []}

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    monkeypatch.setattr(custom_service, "predict_remote_ocr", remote_ocr)
    client = TestClient(custom_service.app)

    response = client.post(
        "/api/v1/ocr-results",
        files={"image": ("sample.png", buffer.getvalue(), "image/png")},
        data={
            "version": "6",
            "model": "baseline",
            "det_model": "thai_det",
            "rec_model": "thai_rec",
        },
    )

    assert response.status_code == 200
    assert captured["version"] == "v6"
    assert captured["model"] == "baseline"
    assert captured["det_model"] == "thai_det"
    assert captured["rec_model"] == "thai_rec"
