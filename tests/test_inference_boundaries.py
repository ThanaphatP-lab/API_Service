import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFERENCE_MODULES = (
    "layout_detection.py",
    "text_detection.py",
    "text_recognition.py",
    "table_structure.py",
    "paddle_ocr.py",
    "table_recognition_v2.py",
    "siglip.py",
)
MIGRATED_SERVICES = (
    "layout/main.py",
    "text_det/main.py",
    "text_rec/main.py",
    "table/main.py",
    "ocr_pipeline_paddle/main.py",
    "table_v2/main.py",
    "siglip/main.py",
)


def _tree(relative_path: str) -> ast.Module:
    path = ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_functions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_inference_modules_expose_consistent_core_interface():
    for filename in INFERENCE_MODULES:
        functions = _top_level_functions(_tree(f"inference/{filename}"))
        assert {"selection_from_settings", "get_model", "infer"} <= functions


def test_migrated_http_services_do_not_import_model_frameworks_directly():
    forbidden = {"paddleocr", "torch", "transformers"}
    for relative_path in MIGRATED_SERVICES:
        tree = _tree(f"services/{relative_path}")
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        assert not (forbidden & imported_roots), relative_path
        assert "inference" in imported_roots, relative_path


def test_legacy_model_wrappers_removed():
    for filename in ("det_model.py", "rec_model.py", "table_v2_model.py"):
        assert not (ROOT / "models" / filename).exists()
