import json

import pytest

from shared.inference_adapters import (
    CONTRACT_VERSION,
    adapt_layout,
    adapt_siglip,
    adapt_table,
    adapt_table_structure,
    adapt_text_detection,
    adapt_text_detection_batch,
    adapt_text_recognition,
    adapt_text_recognition_batch,
)
from shared.serialization import to_jsonable


class FakePaddleResult:
    def __init__(self, result):
        self.res = result


def _assert_json(payload):
    # allow_nan=False verifies that the adapter returns strict JSON values.
    json.dumps(payload, allow_nan=False)
    assert payload["contract_version"] == CONTRACT_VERSION


def test_layout_adapter_matches_notebook_contract_and_preserves_raw_alias():
    payload = adapt_layout(
        [
            FakePaddleResult(
                {
                    "boxes": [
                        {
                            "coordinate": [1, 2, 30, 40],
                            "label": "table",
                            "score": 0.9,
                            "cls_id": 3,
                        }
                    ]
                }
            )
        ]
    )

    _assert_json(payload)
    assert payload["result"] == {
        "detections": [
            {
                "bbox": [1, 2, 30, 40],
                "label": "table",
                "score": 0.9,
                "cls_id": 3,
                "polygon_points": None,
            }
        ]
    }
    assert payload["predictions"] == payload["raw_output"]
def test_text_detection_single_and_batch_keep_one_result_per_image():
    first = {"dt_polys": [[[1, 2], [3, 4]]], "dt_scores": [float("nan")]}
    second = {"dt_polys": [], "dt_scores": []}

    single = adapt_text_detection([first])
    batch = adapt_text_detection_batch([first, second], expected_count=3)

    _assert_json(single)
    _assert_json(batch)
    assert single["result"] == {"dt_polys": [[[1, 2], [3, 4]]], "dt_scores": [None]}
    assert batch["results"][0] == single["result"]
    assert batch["results"][1] == {"dt_polys": [], "dt_scores": []}
    assert batch["results"][2] == {"dt_polys": [], "dt_scores": []}
    assert batch["predictions"] == batch["results"]
    assert batch["count"] == 3


def test_recognition_adapters_match_notebook_contract():
    single = adapt_text_recognition([{"rec_text": "ทดสอบ", "rec_score": 0.75}])
    batch = adapt_text_recognition_batch(
        [{"text": "A", "confidence": 0.8}, {"label": "B", "score": 0.7}],
        expected_count=3,
    )

    _assert_json(single)
    _assert_json(batch)
    assert single["result"] == {"rec_text": "ทดสอบ", "rec_score": 0.75}
    assert batch["results"] == [
        {"rec_text": "A", "rec_score": 0.8},
        {"rec_text": "B", "rec_score": 0.7},
        {"rec_text": "", "rec_score": None},
    ]


def test_table_and_siglip_expose_notebook_result_without_business_logic():
    table = adapt_table([FakePaddleResult({"table_res_list": [{"html": "<table></table>"}]})])
    wired = adapt_table_structure([FakePaddleResult({"structure": ["<table>", "</table>"]})])
    siglip = adapt_siglip(logits=[[1.2, -0.4]], categories=["qr", "other"], device="cuda:0")

    _assert_json(table)
    _assert_json(wired)
    _assert_json(siglip)
    assert table["kind"] == "table_recognition_pipeline_v2"
    assert table["result"] == {"raw_output": table["raw_output"]}
    assert wired["kind"] == "table_structure"
    assert wired["result"] == {"raw_output": wired["raw_output"]}
    assert siglip["result"] == {"logits": [1.2, -0.4], "categories": ["qr", "other"]}
    assert siglip["device"] == "cuda:0"


def test_json_serializer_removes_nested_non_finite_values():
    np = pytest.importorskip("numpy")
    converted = to_jsonable(np.array([np.nan, np.inf, -np.inf, 1.0]))

    assert converted == [None, None, None, 1.0]
    json.dumps(converted, allow_nan=False)
