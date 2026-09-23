import json

import pytest

from shared.contracts import ModelAPIError
from shared.model_variants import (
    clear_model_variant_config_cache,
    normalize_model_version,
    resolve_ocr_model_pair,
    resolve_model_variant,
)


@pytest.fixture(autouse=True)
def clear_variant_cache():
    clear_model_variant_config_cache()
    yield
    clear_model_variant_config_cache()


def test_version_accepts_numeric_and_prefixed_forms():
    assert normalize_model_version("5") == "v5"
    assert normalize_model_version("v5") == "v5"
    assert normalize_model_version("6") == "v6"
    assert normalize_model_version("V6") == "v6"


def test_baseline_remains_available_without_local_weights(monkeypatch):
    monkeypatch.delenv("DET_MODEL_NAME", raising=False)
    monkeypatch.delenv("DET_MODEL_DIR", raising=False)
    monkeypatch.setenv("DET_MODEL_VERSION", "v5")

    selected = resolve_model_variant("detection", "5", "baseline")

    assert selected.version == "v5"
    assert selected.variant == "baseline"
    assert selected.model_name == "PP-OCRv5_server_det"
    assert selected.model_dir is None


def test_integrated_ocr_pair_resolves_detection_and_recognition():
    selected = resolve_ocr_model_pair("5", "baseline")

    assert selected.version == "v5"
    assert selected.variant == "baseline"
    assert selected.detection.kind == "detection"
    assert selected.recognition.kind == "recognition"


def test_integrated_ocr_pair_supports_independent_variants(tmp_path, monkeypatch):
    recognition_dir = tmp_path / "recognition" / "thai_ft_v1" / "infer"
    recognition_dir.mkdir(parents=True)
    (recognition_dir / "inference.json").write_text("{}", encoding="utf-8")
    config = {
        "detection": {
            "v5": {
                "baseline": {
                    "model_name": "PP-OCRv5_server_det",
                    "requires_local_weights": False,
                }
            }
        },
        "recognition": {
            "v5": {
                "thai_ft_v1": {
                    "model_name": "th_PP-OCRv5_mobile_rec",
                    "model_dir": str(recognition_dir),
                    "requires_local_weights": True,
                }
            }
        },
    }
    config_path = tmp_path / "variants.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("MODEL_VARIANTS_CONFIG", str(config_path))
    monkeypatch.delenv("DET_MODEL_DIR", raising=False)
    monkeypatch.delenv("REC_MODEL_DIR", raising=False)
    clear_model_variant_config_cache()

    selected = resolve_ocr_model_pair(
        "5",
        "baseline",
        detection_variant="baseline",
        recognition_variant="thai_ft_v1",
    )

    assert selected.detection.variant == "baseline"
    assert selected.recognition.variant == "thai_ft_v1"
    assert selected.recognition.model_dir == recognition_dir.resolve()


def test_missing_finetuned_weights_return_clear_error(monkeypatch):
    monkeypatch.delenv("REC_MODEL_DIR", raising=False)

    with pytest.raises(ModelAPIError) as captured:
        resolve_model_variant("recognition", "v6", "thai_ft_v1")

    assert captured.value.status_code == 422
    assert captured.value.code == "MODEL_VARIANT_UNAVAILABLE"


def test_custom_registry_can_add_a_variant_without_code_changes(tmp_path, monkeypatch):
    weight_dir = tmp_path / "weights" / "infer"
    weight_dir.mkdir(parents=True)
    (weight_dir / "inference.json").write_text("{}", encoding="utf-8")
    config = {
        "recognition": {
            "v6": {
                "finetuned_v3": {
                    "model_name": "PP-OCRv6_medium_rec",
                    "model_dir": str(weight_dir),
                    "requires_local_weights": True,
                }
            }
        }
    }
    config_path = tmp_path / "variants.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("MODEL_VARIANTS_CONFIG", str(config_path))
    clear_model_variant_config_cache()

    selected = resolve_model_variant("recognition", "6", "finetuned_v3")

    assert selected.variant == "finetuned_v3"
    assert selected.model_dir == weight_dir.resolve()
