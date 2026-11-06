from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.serialization import prediction_list, to_jsonable


CONTRACT_VERSION = "leaf-inference-v1"
_IMAGE_FIELDS = frozenset({"input_img", "output_img"})


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        try:
            nested = getattr(value, "res", None)
        except Exception:
            nested = None
        if isinstance(nested, Mapping):
            result = dict(nested)
        else:
            converted = to_jsonable(value)
            result = dict(converted) if isinstance(converted, Mapping) else {}
    nested = result.get("res")
    return dict(nested) if isinstance(nested, Mapping) else result


def _base(kind: str) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "kind": kind,
    }


def adapt_layout(output: Any) -> dict[str, Any]:
    raw = prediction_list(output, exclude_keys=_IMAGE_FIELDS)
    return {**_base("layout"), "raw_output": raw}


def layout_detections(raw: list[Any]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for obj in raw:
        item = mapping(obj)
        boxes = item.get("boxes")
        layout_result = item.get("layout_det_res")
        if boxes is None and isinstance(layout_result, Mapping):
            boxes = layout_result.get("boxes")
        boxes = to_jsonable(boxes)
        if isinstance(boxes, list):
            for box in boxes:
                if not isinstance(box, Mapping):
                    continue
                detections.append(
                    {
                        "bbox": to_jsonable(first_not_none(box.get("bbox"), box.get("coordinate"), box.get("box"))),
                        "label": to_jsonable(first_not_none(box.get("label"), box.get("category"), box.get("type"))),
                        "score": to_jsonable(first_not_none(box.get("score"), box.get("confidence"))),
                        "cls_id": to_jsonable(box.get("cls_id")),
                        "polygon_points": to_jsonable(box.get("polygon_points")),
                    }
                )
        else:
            bbox = first_not_none(item.get("bbox"), item.get("coordinate"), item.get("box"))
            label = first_not_none(item.get("label"), item.get("category"), item.get("type"))
            if bbox is not None or label is not None:
                detections.append(
                    {
                        "bbox": to_jsonable(bbox),
                        "label": to_jsonable(label),
                        "score": to_jsonable(first_not_none(item.get("score"), item.get("confidence"))),
                        "cls_id": to_jsonable(item.get("cls_id")),
                        "polygon_points": to_jsonable(item.get("polygon_points")),
                    }
                )
    return detections


def _text_detection_result(raw: list[Any]) -> dict[str, Any]:
    polygons: list[Any] = []
    scores: list[Any] = []
    for obj in raw:
        item = mapping(obj)
        polygon_value = first_not_none(item.get("dt_polys"), item.get("polys"), item.get("polygons"), item.get("boxes"))
        score_value = first_not_none(item.get("dt_scores"), item.get("scores"), item.get("confidence"))
        polygon_value = to_jsonable(polygon_value)
        if isinstance(polygon_value, list) and polygon_value:
            single = isinstance(polygon_value[0], list) and bool(polygon_value[0]) and isinstance(polygon_value[0][0], (int, float))
            if single:
                polygons.append(polygon_value)
            else:
                polygons.extend(poly for poly in polygon_value if poly is not None)
        score_value = to_jsonable(score_value)
        if isinstance(score_value, list):
            scores.extend(score_value)
        elif score_value is not None:
            scores.append(score_value)
    return {"dt_polys": polygons, "dt_scores": scores}


def adapt_text_detection(output: Any) -> dict[str, Any]:
    raw = prediction_list(output, exclude_keys=_IMAGE_FIELDS)
    return {**_base("text_detection"), "raw_output": raw}


def adapt_text_detection_batch(output: Any, expected_count: int | None = None) -> dict[str, Any]:
    raw = prediction_list(output, exclude_keys=_IMAGE_FIELDS)
    if expected_count is not None:
        raw = (raw + [{}] * max(0, expected_count - len(raw)))[:expected_count]
    return {**_base("text_detection_batch"), "raw_output": raw, "count": len(raw)}


def recognition_result(value: Any) -> dict[str, Any]:
    item = mapping(value)
    text = first_not_none(item.get("rec_text"), item.get("text"), item.get("label"))
    score = first_not_none(item.get("rec_score"), item.get("score"), item.get("confidence"))
    return {"rec_text": "" if text is None else str(to_jsonable(text)), "rec_score": to_jsonable(score)}


def adapt_text_recognition(output: Any) -> dict[str, Any]:
    raw = prediction_list(output, exclude_keys=_IMAGE_FIELDS)
    return {**_base("text_recognition"), "raw_output": raw}


def adapt_text_recognition_batch(output: Any, expected_count: int | None = None) -> dict[str, Any]:
    raw = prediction_list(output, exclude_keys=_IMAGE_FIELDS)
    if expected_count is not None:
        raw = (raw + [{}] * max(0, expected_count - len(raw)))[:expected_count]
    return {**_base("text_recognition_batch"), "raw_output": raw, "count": len(raw)}


def adapt_table(output: Any) -> dict[str, Any]:
    raw = prediction_list(output, exclude_keys=_IMAGE_FIELDS)
    return {**_base("table_recognition_pipeline_v2"), "raw_output": raw}


def adapt_table_structure(output: Any) -> dict[str, Any]:
    raw = prediction_list(output)
    return {**_base("table_structure"), "raw_output": raw}


def adapt_siglip(*, logits: Any, categories: list[str], device: str) -> dict[str, Any]:
    values = to_jsonable(logits)
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    values = values if isinstance(values, list) else []
    return {**_base("siglip"), "raw_output": [{"logits_per_image": values}],
            "categories": list(categories), "device": device}
