import pytest

from pipelines.verification.scoring import normalize_categories, verify_classification, verify_classification_targets


def test_verification_preserves_original_binary_top_rank_rule():
    categories = normalize_categories(
        [
            {"value": "qr", "label": "QR", "prompt": "qr prompt", "match_threshold": 0.5, "margin_threshold": 0.1},
            {"value": "bar", "label": "Barcode", "prompt": "bar prompt", "match_threshold": 0.5, "margin_threshold": 0.1},
        ]
    )
    result = verify_classification(
        target_value="qr",
        categories=categories,
        classification={
            "predictions": [
                {"label": "qr prompt", "score": 0.9, "logit": 3.0},
                {"label": "bar prompt", "score": 0.4, "logit": 1.0},
            ]
        },
    )

    assert result["passed"] is True
    assert result["target_rank"] == 1
    assert result["score"] == 1.0
    assert result["evidence_score"] == 1.0
    assert result["score_margin"] == pytest.approx(2.0)
    assert result["labels"][0]["raw_logit"] == pytest.approx(3.0)


def test_verification_does_not_reject_top_rank_for_new_margin_threshold():
    categories = normalize_categories(
        [
            {"value": "qr", "prompt": "qr prompt", "match_threshold": 0.5, "margin_threshold": 0.1},
            {"value": "bar", "prompt": "bar prompt", "match_threshold": 0.5, "margin_threshold": 0.1},
        ]
    )
    result = verify_classification(
        target_value="qr",
        categories=categories,
        classification={
            "predictions": [
                {"label": "qr prompt", "score": 0.55, "logit": 0.01},
                {"label": "bar prompt", "score": 0.51, "logit": 0.0},
            ]
        },
    )

    assert result["passed"] is True
    assert result["failure_reason"] == "passed"


def test_verification_fails_when_target_is_not_top_ranked():
    categories = normalize_categories(
        [
            {"value": "qr", "prompt": "qr prompt"},
            {"value": "bar", "prompt": "bar prompt"},
        ]
    )
    result = verify_classification(
        target_value="qr",
        categories=categories,
        classification={
            "predictions": [
                {"label": "bar prompt", "score": 0.9, "logit": 2.0},
                {"label": "qr prompt", "score": 0.8, "logit": 1.0},
            ]
        },
    )

    assert result["passed"] is False
    assert result["score"] == 0.0
    assert result["failure_reason"] == "predicted_category_mismatch"


def test_multiple_targets_reuse_one_classification_payload():
    categories = normalize_categories(
        [
            {"value": "qr", "prompt": "qr prompt"},
            {"value": "bar", "prompt": "bar prompt"},
        ]
    )
    classification = {
        "predictions": [
            {"label": "qr prompt", "score": 0.9, "logit": 3.0},
            {"label": "bar prompt", "score": 0.4, "logit": 1.0},
        ]
    }
    results = verify_classification_targets(
        target_values=["qr", "bar"], categories=categories, classification=classification
    )
    assert [item["image_category"] for item in results] == ["qr", "bar"]
    assert [item["passed"] for item in results] == [True, False]
    assert [item["raw_logit"] for item in results[0]["labels"]] == [
        item["raw_logit"] for item in results[1]["labels"]
    ]
