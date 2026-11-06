import importlib.util
import sys
import types
from pathlib import Path

import pytest
from PIL import Image
from pipelines.layout.contracts import format_legacy_detection


@pytest.mark.parametrize("module_name,model_class,single,batch,item,expected", [
    ("text_detection", "TextDetection", "predict_detection", "predict_detection_batch",
     {"dt_polys": [[[1, 2], [3, 4]]], "dt_scores": [0.9]},
     {"dt_polys": [[[1, 2], [3, 4]]], "dt_scores": [0.9]}),
    ("text_recognition", "TextRecognition", "predict_recognition", "predict_recognition_batch",
     {"text": "A", "confidence": 0.8}, {"rec_text": "A", "rec_score": 0.8}),
])
def test_compatibility_functions_read_raw_output(monkeypatch, module_name, model_class, single, batch, item, expected):
    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(**{model_class: object}))
    path = Path(__file__).parents[1] / "inference" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"compat_test_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "infer_detection" if module_name == "text_detection" else "infer_recognition",
                        lambda *args: {"raw_output": [item]})
    monkeypatch.setattr(module, "infer_detection_batch" if module_name == "text_detection" else "infer_recognition_batch",
                        lambda *args: {"raw_output": [item, item]})
    assert getattr(module, single)("one.png") == [expected]
    assert getattr(module, batch)(["one.png", "two.png"]) == [expected, expected]


def test_legacy_detection_reads_raw_output_per_image(tmp_path):
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in paths:
        Image.new("RGB", (20, 20)).save(path)
    item = {"dt_polys": [[[1, 2], [10, 2], [10, 8], [1, 8]]], "dt_scores": [0.9]}
    result = format_legacy_detection({"raw_output": [item, {}]}, paths, multiple=True)
    assert len(result["results"][0]["regions"]) == 1
    assert result["results"][1]["regions"] == []


def test_verification_uses_siglip_raw_output():
    from shared.inference_adapters import adapt_siglip
    from pipelines.verification.scoring import normalize_categories, verify_classification
    categories = normalize_categories([{"value": "qr", "prompt": "qr"}, {"value": "bar", "prompt": "bar"}])
    classification = adapt_siglip(logits=[3.0, -1.0], categories=["qr", "bar"], device="cpu")
    result = verify_classification(target_value="qr", categories=categories, classification=classification)
    assert result["passed"] is True
    assert result["labels"][0]["raw_logit"] == 3.0
