import asyncio
import base64
import importlib.util
import io
import sys
import threading
import types
from pathlib import Path

import httpx
from PIL import Image

from shared.inference_adapters import adapt_text_recognition, adapt_text_recognition_batch


def load_service(monkeypatch, *, batch_handler, single_handler=None, loader=None):
    fake = types.ModuleType("inference.text_recognition")
    fake.get_model = loader or (lambda *args: None)
    fake.infer_batch = batch_handler
    fake.infer = single_handler or (lambda *args: adapt_text_recognition([{"rec_text": "single", "rec_score": 0.9}]))
    monkeypatch.setitem(sys.modules, "inference.text_recognition", fake)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "0")
    monkeypatch.setenv("MAX_BATCH_IMAGES", "128")
    monkeypatch.setenv("REC_FAIR_CHUNK_SIZE", "20")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "1")  # service-specific admission overrides this
    path = Path(__file__).resolve().parents[1] / "services/text_rec/main.py"
    spec = importlib.util.spec_from_file_location("rec_api_test", path)
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    return service


def encoded_image():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_concurrent_http_batches_preserve_selection_order_and_schema(monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls, paths_used = [], []
    counts = {"v5": 0, "v6": 0}
    def infer(paths, selected):
        paths_used.extend(paths)
        assert all(Path(path).exists() for path in paths)
        calls.append((selected.version, len(paths), threading.get_ident()))
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        offset = counts[selected.version]
        counts[selected.version] += len(paths)
        payload = adapt_text_recognition_batch([
            {"rec_text": f"{selected.version}-{offset+i}", "rec_score": 0.9}
            for i in range(len(paths))
        ], expected_count=len(paths))
        payload["model_selection"] = selected.public_dict()
        return payload
    service = load_service(monkeypatch, batch_handler=infer)
    async def scenario():
        async with service.lifespan(service.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app), base_url="http://test") as client:
                data = encoded_image()
                a = asyncio.create_task(client.post("/api/v1/text-recognition-batches?version=5", json={"images": [data]*100}))
                try:
                    assert await asyncio.to_thread(started.wait, 3)
                    # The event loop remains responsive while native inference blocks.
                    assert (await asyncio.wait_for(client.get("/health"), 1)).status_code == 200
                    b = asyncio.create_task(client.post("/api/v1/text-recognition-batches?version=6", json={"images": [data]*60}))
                    async def admitted():
                        while len(service.app.state.rec_queue._jobs) < 2:
                            await asyncio.sleep(0.001)
                    await asyncio.wait_for(admitted(), 3)
                finally:
                    release.set()
                first, second = await asyncio.gather(a, b)
                for response, version, count in [(first, "v5", 100), (second, "v6", 60)]:
                    assert response.status_code == 200
                    body = response.json()["data"]
                    assert body["count"] == count
                    assert body["model_selection"]["version"] == version
                    assert body["predictions"] == body["results"] == body["result"]["results"]
                    assert [item["rec_text"] for item in body["results"]] == [f"{version}-{i}" for i in range(count)]
                    assert len(body["raw_output"]) == count
    asyncio.run(scenario())
    assert [(version, count) for version, count, _ in calls] == [
        ("v5",20),("v6",20),("v5",20),("v6",20),("v5",20),("v6",20),("v5",20),("v5",20),
    ]
    assert len({thread for _, _, thread in calls}) == 1
    assert not any(Path(path).exists() for path in paths_used)


def test_single_and_readiness_use_same_worker(monkeypatch):
    threads = []
    def single(path, selected):
        threads.append(threading.get_ident())
        payload = adapt_text_recognition([{"rec_text": "single", "rec_score": 0.9}])
        payload["model_selection"] = selected.public_dict()
        return payload
    service = load_service(monkeypatch, batch_handler=lambda *args: None, single_handler=single,
                           loader=lambda *args: threads.append(threading.get_ident()))
    async def scenario():
        async with service.lifespan(service.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app), base_url="http://test") as client:
                response = await client.post("/api/v1/text-recognitions", json={"image": encoded_image()})
                assert response.status_code == 200
                assert response.json()["data"]["rec_text"] == "single"
                assert (await client.get("/api/v1/readiness")).status_code == 200
    asyncio.run(scenario())
    assert len(threads) == 2 and len(set(threads)) == 1
    assert threads[0] != threading.get_ident()
