import pytest

from shared.siglip_categories import siglip_prompts


def test_notebook_category_objects_use_only_enabled_prompts():
    value = [
        {
            "value": "qr_code",
            "label": "QR Code",
            "prompt": "a QR code",
            "match_threshold": 0.55,
            "margin_threshold": 0.05,
            "evidence_temperature": 1.0,
            "enabled": True,
        },
        {"value": "document", "prompt": "a document", "enabled": False},
    ]

    assert siglip_prompts(value) == ["a QR code"]


def test_multipart_json_and_legacy_labels_remain_supported():
    encoded = '[{"value":"qr","prompt":"a QR code","enabled":true}]'

    assert siglip_prompts(encoded) == ["a QR code"]
    assert siglip_prompts(["a QR code", "a document"]) == ["a QR code", "a document"]
    assert siglip_prompts("a QR code,a document") == ["a QR code", "a document"]


def test_enabled_categories_require_prompt_and_boolean_enabled():
    with pytest.raises(ValueError, match="prompt is required"):
        siglip_prompts([{"value": "qr", "enabled": True}])

    with pytest.raises(ValueError, match="must be a boolean"):
        siglip_prompts([{"value": "qr", "prompt": "a QR code", "enabled": "sometimes"}])


def test_disabled_categories_can_omit_prompt():
    assert siglip_prompts([{"value": "unused", "enabled": False}]) == []
