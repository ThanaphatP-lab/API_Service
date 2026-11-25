import hashlib
import io
import json
from pathlib import Path
import runpy

import pytest

from inference.openthai.settings import MODEL_REPOSITORY, MODEL_REVISION, MODEL_VERSION, Settings

ROOT = Path(__file__).resolve().parents[1]


def test_release_metadata_and_secondary_recognizer():
    manifest = json.loads((ROOT / "inference/openthai/model-manifest.json").read_text())
    assert MODEL_REPOSITORY == "iapp/OpenThai2.1-OCR-Text-20261118"
    assert MODEL_VERSION == "2.1.4"
    assert MODEL_REVISION == manifest["revision"]
    assert len(MODEL_REVISION) == 40
    assert "rec2.onnx" in manifest["sha256"]
    assert Settings(_env_file=None).model_dir == ROOT / "weights/openthai2.1"


def test_provision_checks_every_file_before_writing(monkeypatch, tmp_path):
    namespace = runpy.run_path(str(ROOT / "scripts/download-openthai.py"))
    manifest = {"repository": "test/repo", "revision": "pinned", "version": "test",
                "sha256": {"a": hashlib.sha256(b"valid").hexdigest(),
                           "runtime/b": hashlib.sha256(b"expected").hexdigest()}}
    def fetch(url, timeout):
        return io.BytesIO(b"valid" if url.endswith("/a") else b"wrong")
    monkeypatch.setattr("urllib.request.urlopen", fetch)
    output = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="Downloaded checksum mismatch"):
        namespace["provision"](output, manifest)
    assert not output.exists()


def test_provision_reuses_verified_files_without_network(monkeypatch, tmp_path):
    namespace = runpy.run_path(str(ROOT / "scripts/download-openthai.py"))
    (tmp_path / "a").write_bytes(b"valid")
    manifest = {"repository": "test/repo", "revision": "pinned", "version": "test",
                "sha256": {"a": hashlib.sha256(b"valid").hexdigest()}}
    def no_network(*args, **kwargs):
        pytest.fail("Verified local files must not be downloaded again")
    monkeypatch.setattr("urllib.request.urlopen", no_network)
    namespace["provision"](tmp_path, manifest)
    assert json.loads((tmp_path / "provisioned.json").read_text()) == manifest
    (tmp_path / "a").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Existing file"):
        namespace["provision"](tmp_path, manifest)


def test_loader_uses_verified_package_and_requires_rec2(monkeypatch, tmp_path):
    import sys
    import types
    from inference.openthai import engine
    files = {
        "runtime/__init__.py": b"",
        "runtime/thai_norm.py": b"VALUE = 21",
        "runtime/line_engine.py": (
            b"from .thai_norm import VALUE\n"
            b"class LineOCR:\n"
            b"    def __init__(self, directory, threads, runtime):\n"
            b"        self.version = VALUE\n"
            b"        self.backend = runtime['backend']\n"
        ),
        "rec2.onnx": b"test-rec2",
    }
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    manifest = {"revision": "test21", "sha256": {
        name: hashlib.sha256(content).hexdigest() for name, content in files.items()
    }}
    (tmp_path / "model-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(engine, "__file__", str(tmp_path / "engine.py"))
    monkeypatch.setitem(sys.modules, "runtime", types.ModuleType("runtime"))
    inference = engine.Inference(Settings(_env_file=None, model_dir=tmp_path))
    try:
        reader = inference.load()
        assert reader.version == 21
        assert reader.backend == "onnxruntime"
        (tmp_path / "rec2.onnx").unlink()
        with pytest.raises(FileNotFoundError):
            inference.load()
    finally:
        inference.executor.shutdown()
