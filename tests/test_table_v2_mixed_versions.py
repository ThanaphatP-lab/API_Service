import importlib.util
import sys
import types
from pathlib import Path


def test_table_v2_selects_independent_versions_and_applies_baseline_overrides(monkeypatch):
    fake = types.ModuleType("paddleocr")
    fake.TableRecognitionPipelineV2 = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    spec = importlib.util.spec_from_file_location(
        "test_table_v2_inference", Path(__file__).parents[1] / "inference/table_recognition_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from shared import model_variants as registry

    def resolve(kind, version, variant):
        return registry.ModelVariantSpec(kind, version, variant, f"{kind}-{version}", None, False)

    monkeypatch.setattr(registry, "resolve_model_variant", resolve)
    monkeypatch.setenv("TABLE_V2_OCR_VERSION", "v5")
    monkeypatch.setenv("TABLE_V2_TEXT_RECOGNITION_MODEL_NAME", "rec-override")
    selected = module.selection_from_settings(
        detection_version="6", recognition_version="5", recognition_model="baseline"
    )
    assert selected.detection.version == "v6"
    assert selected.recognition.version == "v5"
    options = module.get_model(selected)
    assert options["text_detection_model_name"] == "detection-v6"
    assert options["text_recognition_model_name"] == "rec-override"
