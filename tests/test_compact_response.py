from shared.compact_response import compact_response
from shared.inference_adapters import (adapt_layout, adapt_table, adapt_text_detection,
    adapt_text_detection_batch, adapt_text_recognition, adapt_text_recognition_batch)


def test_compact_single_and_batch_preserve_results_and_selection():
    for adapter, value in [(adapt_text_detection, {"dt_polys": [[[1, 2], [3, 4]]], "dt_scores": [0.9]}),
                           (adapt_text_recognition, {"rec_text": "A", "rec_score": 0.9})]:
        payload = adapter([value])
        payload["model_selection"] = {"version": "v5"}
        assert compact_response(payload, "compact") == {**value, "model_selection": {"version": "v5"}}
        assert compact_response(payload, None) is payload
    for adapter in (adapt_text_detection_batch, adapt_text_recognition_batch):
        payload = adapter([{}], expected_count=2)
        assert len(compact_response(payload, "compact")["results"]) == 2


def test_compact_layout_table_and_legacy():
    layout = adapt_layout([{"boxes": [{"coordinate": [1, 2, 3, 4], "label": "table"}]}])
    assert compact_response({"results": [layout], "count": 1}, "compact") == {
        "results": [compact_response(layout, "compact")], "count": 1}
    table = adapt_table([{"table_res_list": [{"pred_html": "<table></table>",
        "cell_box_list": [[1, 2, 3, 4]], "table_ocr_pred": {"rec_texts": ["A"]},
        "neighbor_texts": "unused", "table_region_id": 1}], "input_img": [1]}])
    assert compact_response(table, "compact") == {"tables": [{"pred_html": "<table></table>",
        "cell_box_list": [[1, 2, 3, 4]], "table_ocr_pred": {"rec_texts": ["A"]}, "table_region_id": 1}]}
    legacy = {"image_width": 20, "regions": [], "raw_predictions": [1]}
    assert compact_response(legacy, "compact") == {"image_width": 20, "regions": []}


def test_layout_pipeline_reads_normalized_layout_and_detection(monkeypatch):
    import numpy as np
    from pathlib import Path
    from pipelines.layout import orchestrator

    monkeypatch.setattr(orchestrator.cv2, "imread", lambda _: np.zeros((100, 100, 3)))
    class Client:
        def infer(self, url, endpoint, paths, **kwargs):
            if endpoint.endswith("layout-predictions"):
                return adapt_layout([{"boxes": [{"coordinate": [10, 10, 50, 50], "label": "table", "score": 0.9}]}])
            return adapt_text_detection([{"dt_polys": [], "dt_scores": []}])
    result = orchestrator.analyze_document_layout(Path("test.png"), layout_url="layout", detector_url="det",
        request_id="test", client=Client(), expand_text_rois=False, auto_roi_mode="layout", padding=(0, 0, 0, 0))
    assert len(result["regions"]) == 1
    assert result["regions"][0]["label"] == "table"


def test_gateway_forwards_compact_query_to_single_and_batch(monkeypatch):
    import base64
    import io
    from PIL import Image
    from fastapi.testclient import TestClient
    from services.gateway import main as gateway

    image = io.BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode("ascii")
    captured = []
    def post(*args, **kwargs):
        captured.append(kwargs["fields"])
        return {"results": [], "count": 0}
    monkeypatch.setattr(gateway, "post_images", post)
    client = TestClient(gateway.app)
    for endpoint, body in [("text-recognitions", {"image": encoded}),
                           ("text-recognition-batches", {"images": [encoded]})]:
        response = client.post(f"/paddle/api/v1/{endpoint}?response_format=compact", json=body)
        assert response.status_code == 200
        assert response.json()["data"] == {"results": [], "count": 0}
    assert [fields["response_format"] for fields in captured] == ["compact", "compact"]
