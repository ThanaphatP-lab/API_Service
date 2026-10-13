import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
from shared import model_variants as variants
from shared.contracts import ModelAPIError


@pytest.mark.parametrize("shape", [None, [3, 48, 320], [3, 48, 640]])
def test_shape_resolution(monkeypatch, shape):
    monkeypatch.setattr(variants, "_load_config", lambda _: {
        "rec_input_shape": shape,
        "recognition": {"v5": {"baseline": {"model_name": "test"}}}})
    selected = variants.resolve_model_variant("recognition", "5")
    assert selected.input_shape == (tuple(shape) if shape else None)


@pytest.mark.parametrize("shape", [[3, 48], [3, 0, 320], [True, 48, 320], "3,48,320", [3, 48, 320.0]])
def test_invalid_shape(monkeypatch, shape):
    monkeypatch.setattr(variants, "_load_config", lambda _: {
        "rec_input_shape": shape,
        "recognition": {"v5": {"baseline": {"model_name": "test"}}}})
    with pytest.raises(ModelAPIError) as error:
        variants.resolve_model_variant("recognition", "5")
    assert error.value.code == "MODEL_VARIANT_CONFIG_INVALID"


@pytest.mark.parametrize("version", ["v5", "v6"])
@pytest.mark.parametrize("variant", ["baseline", "thai_ft_v1"])
def test_global_applies_to_all_rec_variants_not_det(monkeypatch, version, variant):
    entry = {"model_name": "test", "requires_local_weights": False,
             "input_shape": [3, 32, 100]}
    monkeypatch.setattr(variants, "_load_config", lambda _: {
        "rec_input_shape": [3, 48, 640],
        "recognition": {version: {variant: entry}},
        "detection": {version: {variant: entry}}})
    assert variants.resolve_model_variant("recognition", version, variant).input_shape == (3, 48, 640)
    assert variants.resolve_model_variant("detection", version, variant).input_shape is None


def test_missing_global_preserves_model_default(monkeypatch):
    monkeypatch.setattr(variants, "_load_config", lambda _: {
        "recognition": {"v5": {"baseline": {"model_name": "test"}}}})
    assert variants.resolve_model_variant("recognition", "v5").input_shape is None


def test_constructor_and_cache_include_shape(monkeypatch):
    fake = types.ModuleType("paddleocr")
    calls = []
    def constructor(**kwargs):
        calls.append(kwargs)
        return kwargs
    fake.TextRecognition = constructor
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    spec = importlib.util.spec_from_file_location("rec_shape_test", Path(__file__).parents[1] / "inference/text_recognition.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = variants.ModelVariantSpec("recognition", "v5", "baseline", "test", None, False)
    assert "input_shape" not in module.get_model(selected)
    shaped = replace(selected, input_shape=(3, 48, 320))
    assert module.get_model(shaped)["input_shape"] == (3, 48, 320)
    assert module.get_model(shaped) is module.get_model(shaped)
    module.get_model(replace(shaped, input_shape=(3, 48, 640)))
    assert len(calls) == 3
