"""Optional compact representations of leaf inference responses."""
from typing import Any
from shared.inference_adapters import _text_detection_result, recognition_result, layout_detections


def compact_response(payload: dict[str, Any], response_format: Any) -> dict[str, Any]:
    if response_format != "compact":
        return payload
    kind = payload.get("kind")
    selection = {"model_selection": payload["model_selection"]} if "model_selection" in payload else {}
    if kind in {"text_detection_batch", "text_recognition_batch"}:
        raw = payload.get("raw_output") or []
        results = ([_text_detection_result([item]) for item in raw] if kind == "text_detection_batch"
                   else [recognition_result(item) for item in raw])
        return {"results": results, "count": payload["count"], **selection}
    if kind in {"text_detection", "text_recognition"}:
        raw = payload.get("raw_output") or []
        result = _text_detection_result(raw) if kind == "text_detection" else recognition_result(raw[0] if raw else {})
        return {**result, **selection}
    if kind == "layout":
        return {"detections": layout_detections(payload.get("raw_output") or [])}
    if kind == "table_recognition_pipeline_v2":
        tables = []
        for page in payload.get("raw_output", []):
            if not isinstance(page, dict):
                continue
            nested = page.get("res", page)
            if not isinstance(nested, dict):
                continue
            for table in nested.get("table_res_list") or []:
                if isinstance(table, dict):
                    tables.append({key: table[key] for key in
                                   ("pred_html", "cell_box_list", "table_ocr_pred", "table_region_id")
                                   if key in table})
        return {"tables": tables, **selection}
    # Layout batches and legacy detection responses do not have a kind.
    if "results" in payload and "count" in payload:
        return {"results": [compact_response(item, response_format) for item in payload["results"]],
                "count": payload["count"]}
    if "regions" in payload:
        return {key: payload[key] for key in ("engine", "model", "image_width", "image_height", "regions")
                if key in payload}
    return payload


def requested_format(request: Any, fields: dict[str, Any]) -> Any:
    return request.query_params.get("response_format", fields.get("response_format"))
