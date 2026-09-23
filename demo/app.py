from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st


@dataclass(frozen=True)
class APIService:
    label: str
    base_url: str
    path: str
    description: str
    form_kind: str = "image"


def endpoint(env_name: str, default: str) -> str:
    return os.getenv(env_name, default).rstrip("/")


GATEWAY_URL = endpoint("GATEWAY_URL", "http://localhost:8080")

DEFAULT_VERIFICATION_CATEGORIES: list[dict[str, Any]] = [
    {"value": "company_logo", "label": "Company logo", "prompt": "This is a photo of a company logo.", "match_threshold": 0.50, "margin_threshold": 0.05, "enabled": True},
    {"value": "official_stamp", "label": "Official stamp", "prompt": "This is a photo of an official ink stamp on a document.", "match_threshold": 0.50, "margin_threshold": 0.05, "enabled": True},
    {"value": "signature", "label": "Signature", "prompt": "This is a photo of a handwritten signature.", "match_threshold": 0.45, "margin_threshold": 0.04, "enabled": True},
    {"value": "qr_code", "label": "QR Code", "prompt": "This is a photo of a QR code.", "match_threshold": 0.55, "margin_threshold": 0.05, "enabled": True},
    {"value": "barcode", "label": "Barcode", "prompt": "This is a photo of a linear barcode.", "match_threshold": 0.55, "margin_threshold": 0.05, "enabled": True},
    {"value": "portrait", "label": "Portrait", "prompt": "This is a portrait photo of a real person.", "match_threshold": 0.45, "margin_threshold": 0.04, "enabled": True},
    {"value": "government_emblem", "label": "Thai Garuda", "prompt": "This is a photo of the Thai Garuda government emblem.", "match_threshold": 0.40, "margin_threshold": 0.03, "enabled": True},
    {"value": "thailand_symbol", "label": "Thailand symbol", "prompt": "This is a photo of a recognizable symbol associated with Thailand.", "match_threshold": 0.03, "margin_threshold": 0.02, "enabled": True},
]

DEFAULT_CLASSIFICATION_CATEGORIES: list[dict[str, Any]] = [
    {"value": "invoice", "label": "Invoice", "prompt": "This is an invoice document.", "match_threshold": 0.50, "margin_threshold": 0.05, "evidence_temperature": 1.0, "enabled": True},
    {"value": "receipt", "label": "Receipt", "prompt": "This is a receipt.", "match_threshold": 0.50, "margin_threshold": 0.05, "evidence_temperature": 1.0, "enabled": True},
    {"value": "identity_card", "label": "Identity card", "prompt": "This is an identity card.", "match_threshold": 0.50, "margin_threshold": 0.05, "evidence_temperature": 1.0, "enabled": True},
]


def auth_headers(service: APIService) -> dict[str, str]:
    variable = "MODEL_GATEWAY_API_KEY" if service.base_url == GATEWAY_URL else "INTERNAL_API_TOKEN"
    token = os.getenv(variable, "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


SERVICES: dict[str, APIService] = {
    "layout_pipeline": APIService("Gateway: document layout", GATEWAY_URL, "/api/v1/document-layouts", "Layout + text detection + ROI filtering", "layout"),
    "text_detection": APIService("Gateway: text detection (configured default)", GATEWAY_URL, "/api/v1/text-detections", "Detector selected by the layout pipeline configuration"),
    "text_detection_v5": APIService("Gateway: PP-OCRv5 detector", GATEWAY_URL, "/api/v1/text-detections?version=v5", "Direct Gateway routing to the V5 leaf model"),
    "text_detection_v6": APIService("Gateway: PP-OCRv6 detector", GATEWAY_URL, "/api/v1/text-detections?version=v6", "Direct Gateway routing to the V6 leaf model"),
    "ocr_custom": APIService("Gateway: custom OCR", GATEWAY_URL, "/api/v1/ocr-results?engine=custom", "Detector -> crop -> batch recognition"),
    "ocr_paddle": APIService("Gateway: PaddleOCR", GATEWAY_URL, "/api/v1/ocr-results?engine=paddle", "Integrated PaddleOCR detection + Thai recognition", "ocr"),
    "table_pipeline": APIService("Gateway: table pipeline", GATEWAY_URL, "/api/v1/table-results", "Adaptive wired/wireless selection + grid analysis + OCR", "table"),
    "table_v2_gateway": APIService("Gateway: TableRecognitionPipelineV2", GATEWAY_URL, "/api/v1/table-model-results", "Notebook (4) TableV2 raw model output", "table_v2"),
    "image_verification": APIService("Gateway: image verification", GATEWAY_URL, "/api/v1/image-verifications", "SigLIP category verification", "verification"),
    "layout": APIService("Leaf: PP-DocLayoutV3", endpoint("LAYOUT_URL", "http://localhost:8001"), "/api/v1/layout-predictions", "Canonical layout model output"),
    "det_v5": APIService("Leaf: PP-OCRv5 detector", endpoint("DET_V5_URL", "http://localhost:8002"), "/api/v1/text-detections", "Canonical PP-OCRv5 detection output"),
    "det_v6": APIService("Leaf: PP-OCRv6 detector", endpoint("DET_V6_URL", "http://localhost:8003"), "/api/v1/text-detections", "Canonical PP-OCRv6 detection output"),
    "rec_th": APIService("Leaf: Thai text recognition", endpoint("REC_TH_URL", "http://localhost:8004"), "/api/v1/text-recognitions", "Use a pre-cropped text image"),
    "table_wired": APIService("Leaf: SLANeXt wired", endpoint("TABLE_WIRED_URL", "http://localhost:8007"), "/api/v1/table-structures", "Raw wired table structure output"),
    "table_wireless": APIService("Leaf: SLANeXt wireless", endpoint("TABLE_WIRELESS_URL", "http://localhost:8008"), "/api/v1/table-structures", "Raw wireless table structure output"),
    "table_v2_leaf": APIService("Leaf: TableRecognitionPipelineV2", endpoint("TABLE_V2_URL", "http://localhost:8013"), "/api/v1/table-model-results", "Notebook (4) TableV2 inference/serialization", "table_v2"),
    "siglip": APIService("Leaf: SigLIP classifier", endpoint("SIGLIP_URL", "http://localhost:8009"), "/api/v1/image-classifications", "Notebook (4) category-object or legacy label input", "classification"),
}


def get_health(service: APIService) -> tuple[bool, str]:
    try:
        response = requests.get(f"{service.base_url}/api/v1/health", timeout=3)
        body = response.json()
        if response.ok:
            data = body.get("data", {})
            return True, str(data.get("device", "online"))
        error = body.get("error", {})
        return False, str(error.get("code") or f"HTTP {response.status_code}")
    except (requests.RequestException, ValueError):
        return False, "offline"


def call_api(service: APIService, upload: Any, fields: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    files = {"image": (upload.name, upload.getvalue(), upload.type or "image/png")}
    form = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        for key, value in fields.items()
        if value is not None
    }
    try:
        response = requests.post(
            f"{service.base_url}{service.path}",
            files=files,
            data=form,
            headers=auth_headers(service),
            timeout=300,
        )
    except requests.RequestException as exc:
        return 0, {"error": {"code": "DEMO_CONNECTION_ERROR", "message": str(exc)}}
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"error": {"code": "INVALID_JSON", "message": response.text}}


def category_editor(label: str, default: list[dict[str, Any]], *, key: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    raw = st.text_area(
        label,
        value=json.dumps(default, ensure_ascii=False, indent=2),
        height=300,
        key=key,
    )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Category JSON is invalid at line {exc.lineno}, column {exc.colno}."
    if not isinstance(value, list):
        return None, "Categories must be a JSON array."
    if not all(isinstance(item, dict) for item in value):
        return None, "Every category must be a JSON object."
    return value, None


st.set_page_config(page_title="Model API v1 Demo", layout="wide")
st.title("Model API v1 Demo")
st.caption("Test Gateway pipelines and leaf inference contracts, including notebook (4) TableV2 and SigLIP inputs.")

with st.sidebar:
    st.header("Service status")
    st.button("Refresh status")
    with ThreadPoolExecutor(max_workers=8) as executor:
        statuses = list(executor.map(get_health, SERVICES.values()))
    for service, (online, message) in zip(SERVICES.values(), statuses):
        (st.success if online else st.error)(f"{service.label}: {message}")

upload = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp"])
selected_key = st.selectbox("API to test", list(SERVICES), format_func=lambda key: SERVICES[key].label)
selected = SERVICES[selected_key]
st.info(f"POST {selected.base_url}{selected.path} - {selected.description}")

fields: dict[str, Any] = {}
validation_error: str | None = None

if selected.form_kind == "layout":
    fields["expand_text_rois"] = st.checkbox("Expand text ROIs", value=False)
    fields["auto_roi_mode"] = st.selectbox("Auto ROI mode", ["text-line", "layout", "hybrid"])
elif selected.form_kind == "ocr":
    fields["text_det_unclip_ratio"] = st.number_input("Unclip ratio", 0.1, 10.0, 2.0)
    fields["text_det_thresh"] = st.number_input("Detection threshold", 0.0, 1.0, 0.25)
    fields["text_det_box_thresh"] = st.number_input("Box threshold", 0.0, 1.0, 0.60)
elif selected.form_kind == "table":
    fields["table_mode"] = st.selectbox("Table mode", ["auto", "wired", "wireless"])
    fields["include_ocr"] = st.checkbox("Include OCR", value=True)
elif selected.form_kind == "table_v2":
    st.caption("TableV2 uses its notebook (4) model configuration and returns JSON-safe raw_output. No pipeline thresholds are sent.")
elif selected.form_kind == "classification":
    input_mode = st.radio("SigLIP input contract", ["Notebook (4) category objects", "Legacy prompt list"])
    if input_mode == "Notebook (4) category objects":
        categories, validation_error = category_editor(
            "Categories JSON",
            DEFAULT_CLASSIFICATION_CATEGORIES,
            key="classification_categories",
        )
        if categories is not None:
            fields["categories"] = categories
    else:
        fields["labels"] = st.text_input("Candidate prompts", "invoice,receipt,identity card")
elif selected.form_kind == "verification":
    use_custom_categories = st.checkbox("Send custom category configuration", value=False)
    if use_custom_categories:
        categories, validation_error = category_editor(
            "Verification categories JSON",
            DEFAULT_VERIFICATION_CATEGORIES,
            key="verification_categories",
        )
        if categories is not None:
            fields["categories"] = categories
            active_values = [str(item.get("value", "")).strip() for item in categories if item.get("enabled", True)]
            active_values = [value for value in active_values if value]
            if active_values:
                fields["image_category"] = st.selectbox("Expected category", active_values)
            else:
                validation_error = "At least one enabled verification category is required."
    else:
        fields["image_category"] = st.selectbox(
            "Expected category",
            [item["value"] for item in DEFAULT_VERIFICATION_CATEGORIES],
        )

if validation_error:
    st.error(validation_error)

if upload is not None:
    left, right = st.columns([1, 2])
    with left:
        st.image(upload, caption=upload.name, use_container_width=True)
    with right:
        st.write(f"File size: {upload.size:,} bytes")
        st.write(f"Request fields: {', '.join(fields) if fields else 'none'}")

submitted = st.button(
    "Test API",
    type="primary",
    disabled=upload is None or validation_error is not None,
)
if submitted:
    with st.spinner(f"Calling {selected.label}..."):
        status, response_body = call_api(selected, upload, fields)
    if 200 <= status < 300:
        st.success(f"Success - HTTP {status}")
    else:
        message = "connection error" if status == 0 else f"HTTP {status}"
        st.error(f"Request failed - {message}")

    data = response_body.get("data") if isinstance(response_body, dict) else None
    canonical = data.get("result") if isinstance(data, dict) else None
    if canonical is not None:
        canonical_tab, response_tab = st.tabs(["Canonical result", "Full API response"])
        with canonical_tab:
            st.json(canonical)
        with response_tab:
            st.json(response_body)
    else:
        st.json(response_body)

    st.download_button(
        "Download JSON",
        json.dumps(response_body, ensure_ascii=False, indent=2),
        file_name=f"{selected_key}_result.json",
        mime="application/json",
    )

st.divider()
st.caption("OpenAPI is available at /docs on each service when API_DOCS_ENABLED is enabled.")
