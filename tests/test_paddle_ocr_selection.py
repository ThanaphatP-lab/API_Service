import importlib.util
import sys
import types
from pathlib import Path


def load_module(monkeypatch):
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCR = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    spec = importlib.util.spec_from_file_location("test_paddle_inference", Path(__file__).parents[1] / "inference/paddle_ocr.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_legacy_selection_and_cache(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setenv("DET_MODEL_NAME", "legacy-det")
    monkeypatch.setenv("REC_MODEL_NAME", "legacy-rec")
    selection = module.selection_from_settings()
    assert selection.pair is None
    assert selection.detection_model_name == "legacy-det"
    assert module.get_model(selection) is module.get_model(selection)


def test_registry_pair_passes_weights_to_constructor(monkeypatch, tmp_path):
    module = load_module(monkeypatch)
    from shared.model_variants import ModelVariantSpec, OCRModelPairSpec
    pair = OCRModelPairSpec("v5", "thai_ft_v1",
        ModelVariantSpec("detection", "v5", "baseline", "det", None, False),
        ModelVariantSpec("recognition", "v5", "thai_ft_v1", "rec", tmp_path, True))
    monkeypatch.setattr(module, "resolve_ocr_model_pair", lambda *a, **kw: pair)
    selection = module.selection_from_settings("5", recognition_model="thai_ft_v1")
    options = module.get_model(selection)
    assert options["text_recognition_model_dir"] == str(tmp_path)
    assert "text_detection_model_dir" not in options
    assert module._payload([], {}, selection)["model_selection"]["recognition"]["variant"] == "thai_ft_v1"


def test_gateway_paddle_forwards_query_selection_single_and_batch(monkeypatch):
    import asyncio
    from starlette.requests import Request
    from services.gateway import main as gateway
    captured = []
    async def forward(request, **kwargs):
        captured.append(kwargs["field_resolver"]({}))
        return {}
    monkeypatch.setattr(gateway, "_forward", forward)
    monkeypatch.setattr(gateway, "_forward_multiple", forward)
    request = Request({"type": "http", "query_string": b"engine=paddle&version=5&rec_model=thai_ft_v1", "headers": []})
    asyncio.run(gateway.ocr_results(request, engine="paddle"))
    asyncio.run(gateway.ocr_result_batches(request))
    assert len(captured) == 2
    assert all(item["version"] == "v5" and item["rec_model"] == "thai_ft_v1" for item in captured)


def test_mixed_versions_resolve_independently(monkeypatch):
    module = load_module(monkeypatch)
    from shared import model_variants as registry
    calls = []
    def resolve(kind, version, variant):
        calls.append((kind, version, variant))
        return registry.ModelVariantSpec(kind, version, variant, f"{kind}-{version}", None, False)
    monkeypatch.setattr(registry, "resolve_model_variant", resolve)
    selected = module.selection_from_settings(detection_version="6", recognition_version="5", recognition_model="thai_ft_v1")
    assert calls == [("detection", "v6", "baseline"), ("recognition", "v5", "thai_ft_v1")]
    assert selected.pair.public_dict()["recognition"]["version"] == "v5"
    assert selected.pair.version == "det-v6__rec-v5"
    options = module.get_model(selected)
    assert options["text_detection_model_name"] == "detection-v6"
    assert options["text_recognition_model_name"] == "recognition-v5"
    calls.clear()
    module.selection_from_settings("6", recognition_version="5")
    assert [call[1] for call in calls] == ["v6", "v5"]


def test_gateway_mixed_versions_and_validation():
    import pytest
    from starlette.requests import Request
    from services.gateway import main as gateway
    from shared.contracts import ModelAPIError
    request = Request({"type": "http", "query_string": b"det_version=6&rec_version=5&rec_model=thai_ft_v1", "headers": []})
    selected = gateway._request_paddle_selection(request, {"rec_version": "6"})
    assert selected["det_version"] == "v6"
    assert selected["rec_version"] == "v5"
    assert selected["rec_model"] == "thai_ft_v1"
    request = Request({"type": "http", "query_string": b"det_version=7", "headers": []})
    with pytest.raises(ModelAPIError):
        gateway._request_paddle_selection(request, {})
    request = Request({"type": "http", "query_string": b"det_version=6&rec_model=thai_ft_v1", "headers": []})
    with pytest.raises(ModelAPIError):
        gateway._request_paddle_selection(request, {})
