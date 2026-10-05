import numpy as np
import pytest

from shared.inference_adapters import adapt_layout, adapt_text_detection, adapt_text_detection_batch
from shared.serialization import to_jsonable


class UnserializableImage(np.ndarray):
    def tolist(self):
        raise AssertionError("Image pixels must not be converted")


@pytest.mark.parametrize("adapter", [adapt_layout, adapt_text_detection, adapt_text_detection_batch])
def test_filter_before_conversion_preserves_wrappers_and_input(adapter):
    image = np.zeros((2, 2, 3)).view(UnserializableImage)
    nested = {"input_img": image, "dt_polys": [[[0, 0], [1, 1]]], "dt_scores": [0.9], "boxes": []}
    source = {"input_img": image, "res": nested, "page_index": 3}
    output = adapter([source])
    raw = output["raw_output"][0]
    assert "input_img" not in raw and "input_img" not in raw["res"]
    assert raw["page_index"] == 3
    assert raw["res"]["dt_polys"] == nested["dt_polys"]
    assert source["input_img"] is image and nested["input_img"] is image


def test_default_serialization_remains_unchanged():
    assert to_jsonable({"input_img": np.array([1])}) == {"input_img": [1]}


def test_layout_timing_and_old_upstream_filter(monkeypatch):
    from pathlib import Path
    from pipelines.layout import orchestrator
    monkeypatch.setattr(orchestrator.cv2, "imread", lambda _: np.zeros((10, 10, 3)))
    ticks = iter([1.0, 1.1, 1.3, 1.6, 1.7])
    monkeypatch.setattr(orchestrator.time, "perf_counter", lambda: next(ticks))
    responses = [
        {"predictions": [{"res": {"boxes": [], "input_img": [123]}, "page_index": 0}]},
        {"predictions": [{"dt_polys": [], "dt_scores": [], "input_img": [123]}]},
    ]
    class Client:
        def __init__(self):
            self.calls = []
        def infer(self, url, endpoint, paths, **kwargs):
            self.calls.append(endpoint)
            return responses[len(self.calls)-1]
    client = Client()
    result = orchestrator.analyze_document_layout(
        Path("test.png"), layout_url="layout", detector_url="det", request_id="req_test",
        client=client, expand_text_rois=False, auto_roi_mode="layout", padding=(0, 0, 0, 0))
    assert client.calls == ["/api/v1/layout-predictions", "/api/v1/text-detections"]
    assert result["timing"] == {"layout_ms": 200.0, "text_detection_ms": 300.0,
        "model_calls_ms": 500.0, "postprocess_ms": 100.0, "pipeline_ms": 700.0}
    assert result["regions"] == []
    assert "input_img" not in result["raw"]["layout"][0]["res"]
    assert "input_img" not in result["raw"]["detection"][0]
    assert responses[0]["predictions"][0]["res"]["input_img"] == [123]
