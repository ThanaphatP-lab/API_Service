import importlib.util
import sys
import types
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pytest


def load(monkeypatch, name):
    spec = importlib.util.spec_from_file_location("batch_test_" + name, Path(__file__).parents[1] / "inference" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_layout_batches_preserve_pages_and_filter_images(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(LayoutDetection=object))
    module = load(monkeypatch, "layout_detection")
    calls = []
    class Model:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return iter([{"boxes": [], "page": i, "input_img": [0]} for i in range(2)])
    monkeypatch.setattr(module, "get_model", lambda _: Model())
    monkeypatch.setenv("LAYOUT_BATCH_SIZE", "2")
    result = module.infer_batch(["a", "b"], module.LayoutSelection("fake", None, "cpu"))
    assert calls[0]["batch_size"] == 2
    assert result["count"] == 2
    assert [r["raw_output"][0]["page"] for r in result["results"]] == [0, 1]
    assert all("input_img" not in r["raw_output"][0] for r in result["results"])
    with pytest.raises(RuntimeError):
        module.infer_batch(["a"], module.LayoutSelection("fake", None, "cpu"))


def test_siglip_real_batch_shape_and_chunking(monkeypatch):
    class Tensor:
        def __init__(self, values): self.values = np.asarray(values)
        def to(self, device): return self
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return self.values.tolist()
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        inference_mode=nullcontext, sigmoid=lambda t: Tensor(1/(1+np.exp(-t.values)))))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(SiglipModel=object, SiglipProcessor=object))
    module = load(monkeypatch, "siglip")
    calls, images = [], []
    class Image:
        closed = False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def convert(self, mode): images.append(self); return self
        def close(self): self.closed = True
    monkeypatch.setattr(module.Image, "open", lambda _: Image())
    def processor(**kwargs):
        calls.append(len(kwargs["images"]))
        return {"pixels": Tensor([[i, -i] for i in range(len(images)-calls[-1], len(images))])}
    runtime = module.SiglipRuntime(processor, lambda **kw: types.SimpleNamespace(logits_per_image=kw["pixels"]), "cpu")
    monkeypatch.setattr(module, "get_model", lambda _: runtime)
    monkeypatch.setenv("SIGLIP_BATCH_SIZE", "2")
    result = module.infer_batch(["a", "b", "c"], ["text", "image"], module.SiglipSelection("fake", None, "cpu"))
    assert calls == [2, 1]
    assert result["count"] == 3
    assert [r["logits"][0] for r in result["results"]] == [0, 1, 2]
    assert all(image.closed for image in images)


@pytest.mark.parametrize("route,upstream", [
    ("document-layout-batches", "LAYOUT_PIPELINE_URL"),
    ("layout-prediction-batches", "LAYOUT_SERVICE_URL"),
    ("image-classification-batches", "SIGLIP_URL"),
])
def test_gateway_batch_routes(monkeypatch, route, upstream):
    from fastapi.testclient import TestClient
    from services.gateway import main as gateway
    captured = {}
    async def forward(request, **kwargs):
        captured.update(kwargs)
        return {"results": [], "count": 0}
    monkeypatch.setattr(gateway, "_forward_multiple", forward)
    assert TestClient(gateway.app).post("/api/v1/" + route).status_code == 200
    assert captured == {"upstream": getattr(gateway, upstream), "endpoint": "/api/v1/" + route}


@pytest.mark.parametrize("service,module_name,endpoint", [
    ("layout", "layout_detection", "layout-prediction-batches"),
    ("siglip", "siglip", "image-classification-batches"),
])
def test_leaf_batch_http_and_cleanup(monkeypatch, service, module_name, endpoint):
    import io
    from PIL import Image
    from fastapi.testclient import TestClient
    paths_seen = []
    def infer_batch(paths, *args):
        paths_seen.extend(Path(p) for p in paths)
        assert all(Path(p).exists() for p in paths)
        return {"results": [{"index": i} for i in range(len(paths))], "count": len(paths)}
    fake = types.ModuleType("inference." + module_name)
    fake.get_model = lambda *a: None
    fake.selection_from_settings = lambda: None
    fake.infer = lambda *a: {}
    fake.infer_batch = infer_batch
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    spec = importlib.util.spec_from_file_location("batch_service_" + service,
        Path(__file__).parents[1] / "services" / service / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    buffer = io.BytesIO()
    Image.new("RGB", (3, 3)).save(buffer, format="PNG")
    files = [("images", (name, buffer.getvalue(), "image/png")) for name in ("a.png", "b.png")]
    client = TestClient(module.app)
    response = client.post("/api/v1/" + endpoint, files=files, data={"labels": '["text", "image"]'})
    assert response.status_code == 200
    assert response.json()["data"]["count"] == 2
    assert all(not p.exists() for p in paths_seen)
    def fail(paths, *args):
        paths_seen.extend(Path(p) for p in paths)
        raise RuntimeError("test failure")
    monkeypatch.setattr(module, "infer_batch", fail)
    assert client.post("/api/v1/" + endpoint, files=files, data={"labels": '["text"]'}).status_code == 500
    assert all(not p.exists() for p in paths_seen)


def test_document_batch_reuses_pipeline_per_page(monkeypatch):
    import io
    from PIL import Image
    from fastapi.testclient import TestClient
    from services.layout_pipeline import main as service
    paths_seen = []
    def analyze(path, **kwargs):
        paths_seen.append(path)
        return {"page": len(paths_seen), "timing": {"pipeline_ms": 1}}
    monkeypatch.setattr(service, "analyze_document_layout", analyze)
    buffer = io.BytesIO()
    Image.new("RGB", (3, 3)).save(buffer, format="PNG")
    response = TestClient(service.app).post("/api/v1/document-layout-batches", files=[
        ("images", ("a.png", buffer.getvalue(), "image/png")),
        ("images", ("b.png", buffer.getvalue(), "image/png")),
    ])
    assert response.status_code == 200
    assert [item["page"] for item in response.json()["data"]["results"]] == [1, 2]
    assert all(not p.exists() for p in paths_seen)
