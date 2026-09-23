from __future__ import annotations

import math
from typing import Any

from shared.contracts import ModelAPIError


DEFAULT_CATEGORIES: list[dict[str, Any]] = [
    {"value": "company_logo", "label": "Company logo", "prompt": "This is a photo of a company logo.", "match_threshold": 0.50, "margin_threshold": 0.05, "enabled": True},
    {"value": "official_stamp", "label": "Official stamp", "prompt": "This is a photo of an official ink stamp on a document.", "match_threshold": 0.50, "margin_threshold": 0.05, "enabled": True},
    {"value": "signature", "label": "Signature", "prompt": "This is a photo of a handwritten signature.", "match_threshold": 0.45, "margin_threshold": 0.04, "enabled": True},
    {"value": "qr_code", "label": "QR Code", "prompt": "This is a photo of a QR code.", "match_threshold": 0.55, "margin_threshold": 0.05, "enabled": True},
    {"value": "barcode", "label": "Barcode", "prompt": "This is a photo of a linear barcode.", "match_threshold": 0.55, "margin_threshold": 0.05, "enabled": True},
    {"value": "portrait", "label": "Portrait", "prompt": "This is a portrait photo of a real person.", "match_threshold": 0.45, "margin_threshold": 0.04, "enabled": True},
    {"value": "government_emblem", "label": "Thai Garuda", "prompt": "This is a photo of the Thai Garuda government emblem.", "match_threshold": 0.40, "margin_threshold": 0.03, "enabled": True},
    {"value": "thailand_symbol", "label": "Thailand symbol", "prompt": "This is a photo of a recognizable symbol associated with Thailand, such as the map of Thailand, the Thai national flag, or a Thai elephant symbol.", "match_threshold": 0.03, "margin_threshold": 0.02, "enabled": True},
]


def _ui_percentages(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = [
        {
            "image_category": item["image_category"],
            "label": item["label"],
            "prompt": item["prompt"],
            "percentage": round(max(0.0, float(item["relative_score"])) * 100.0, 1),
        }
        for item in labels
    ]
    delta = round(100.0 - sum(float(item["percentage"]) for item in values), 1)
    if values:
        best = max(range(len(values)), key=lambda index: values[index]["percentage"])
        values[best]["percentage"] = round(float(values[best]["percentage"]) + delta, 1)
    return values


def normalize_categories(value: Any) -> list[dict[str, Any]]:
    categories = DEFAULT_CATEGORIES if value is None else value
    if not isinstance(categories, list):
        raise ModelAPIError(
            422,
            "VALIDATION_ERROR",
            "categories must be a JSON array.",
            details=[{"field": "categories", "issue": "array_required"}],
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, category in enumerate(categories):
        if not isinstance(category, dict):
            raise ModelAPIError(
                422,
                "VALIDATION_ERROR",
                "Every category must be an object.",
                details=[{"field": f"categories.{index}", "issue": "object_required"}],
            )
        if category.get("enabled", True) is False:
            continue
        value_name = str(category.get("value", "")).strip()
        prompt = str(category.get("prompt", "")).strip()
        if not value_name or not prompt:
            raise ModelAPIError(
                422,
                "VALIDATION_ERROR",
                "Each category requires value and prompt.",
                details=[{"field": f"categories.{index}", "issue": "value_and_prompt_required"}],
            )
        if value_name == "other" or value_name in seen:
            continue
        seen.add(value_name)
        normalized.append(
            {
                "value": value_name,
                "label": str(category.get("label") or value_name),
                "prompt": prompt,
                "match_threshold": max(0.0, min(1.0, float(category.get("match_threshold", 0.70)))),
                "margin_threshold": max(0.0, min(1.0, float(category.get("margin_threshold", 0.05)))),
            }
        )
    if not normalized:
        raise ModelAPIError(422, "NO_ACTIVE_CATEGORIES", "At least one category must be enabled.")
    return normalized


def verify_classification(
    *,
    target_value: str,
    categories: list[dict[str, Any]],
    classification: dict[str, Any],
) -> dict[str, Any]:
    target = next((item for item in categories if item["value"] == target_value), None)
    if target is None:
        raise ModelAPIError(
            422,
            "CATEGORY_NOT_FOUND",
            "image_category is not present in categories.",
            details=[{"field": "image_category", "received": target_value}],
        )
    prompt_map = {item["prompt"]: item for item in categories}
    ranked = []
    for prediction in classification.get("predictions") or []:
        category = prompt_map.get(str(prediction.get("label")))
        if category:
            pair_score = float(prediction.get("score", 0.0))
            raw_logit = prediction.get("logit")
            if raw_logit is None:
                # Compatibility with older/mocked classifier responses which
                # exposed sigmoid probabilities only.
                bounded = min(max(pair_score, 1e-7), 1.0 - 1e-7)
                raw_logit = math.log(bounded / (1.0 - bounded))
            ranked.append({**category, "raw_pair_score": pair_score, "raw_logit": float(raw_logit)})
    if not ranked:
        raise ModelAPIError(502, "CLASSIFICATION_EMPTY", "SigLIP did not return category scores.")
    ranked.sort(key=lambda item: item["raw_logit"], reverse=True)
    max_logit = max(item["raw_logit"] for item in ranked)
    exponentials = [math.exp(item["raw_logit"] - max_logit) for item in ranked]
    exponential_total = sum(exponentials) or 1.0
    for index, item in enumerate(ranked):
        item["relative_score"] = exponentials[index] / exponential_total
    target_rank = next((index + 1 for index, item in enumerate(ranked) if item["value"] == target_value), 0)
    target_result = next(item for item in ranked if item["value"] == target_value)
    top_logit = ranked[0]["raw_logit"]
    second_logit = ranked[1]["raw_logit"] if len(ranked) > 1 else top_logit
    margin = top_logit - second_logit if target_rank == 1 else target_result["raw_logit"] - top_logit
    # Preserve the original binary-v3 rule: the target passing rank 1 is the
    # decision. Thresholds stay in the response as configuration/audit data.
    passed = target_rank == 1
    labels = []
    for rank, item in enumerate(ranked, start=1):
        labels.append(
            {
                "rank": rank,
                "image_category": item["value"],
                "value": item["value"],
                "label": item["label"],
                "prompt": item["prompt"],
                "raw_logit": round(item["raw_logit"], 4),
                "raw_pair_score": round(item["raw_pair_score"], 4),
                "score": round(item["raw_pair_score"], 6),
                "relative_score": round(item["relative_score"], 4),
                "relative_percentage": round(item["relative_score"] * 100.0, 2),
                "field_score": 1.0 if rank == 1 else 0.0,
                "evidence_score": 1.0 if rank == 1 else 0.0,
                "match_threshold": round(float(item["match_threshold"]), 4),
                "margin_threshold": round(float(item["margin_threshold"]), 4),
                "target": item["value"] == target_value,
            }
        )
    top = ranked[0]
    evidence = 1.0 if passed else 0.0
    return {
        "passed": passed,
        "status": "matched" if passed else "mismatched",
        "failure_reason": "passed" if passed else "predicted_category_mismatch",
        "image_category": target_value,
        "image_category_label": target["label"],
        "prompt": target["prompt"],
        "predicted_category": top["value"],
        "predicted_label": top["label"],
        "predicted_prompt": top["prompt"],
        "score": evidence,
        "evidence_score": evidence,
        "target_rank": target_rank,
        "score_margin": round(margin, 6),
        "raw_logit": round(target_result["raw_logit"], 4),
        "raw_pair_score": round(target_result["raw_pair_score"], 4),
        "relative_percentage": round(target_result["relative_score"] * 100.0, 2),
        "verification_threshold": target["match_threshold"],
        "margin_threshold": target["margin_threshold"],
        "model_name": "google/siglip-so400m-patch14-384",
        "model_version": "google/siglip-so400m-patch14-384",
        "scoring_version": "siglip-image-category-binary-v3",
        "version": "siglip-image-category-binary-v3",
        "device": classification.get("device", "remote"),
        "labels": labels,
        "rankings": labels,
        "ui_percentages": _ui_percentages(labels),
        "error": None,
    }


def verify_classification_targets(
    *,
    target_values: list[str],
    categories: list[dict[str, Any]],
    classification: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate several acceptable targets from one shared SigLIP inference."""
    unique_targets = list(dict.fromkeys(str(value).strip() for value in target_values if str(value).strip()))
    return [
        verify_classification(target_value=value, categories=categories, classification=classification)
        for value in unique_targets
    ]
