from __future__ import annotations

import base64
import binascii
import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from fastapi import Request
from PIL import Image, UnidentifiedImageError
from shared.contracts import ModelAPIError
from core.errors import _safe_received
from core.limits import _max_upload_bytes, check_batch_limits
from core.settings import limits_settings
from core.telemetry import timed_stage


IMAGE_REQUEST_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "format": "binary"},
                        "image_base64": {"type": "string", "description": "Optional Data URL/Base64 alternative"},
                    },
                }
            },
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["image"],
                    "properties": {"image": {"type": "string", "description": "Base64 or Data URL"}},
                    "additionalProperties": True,
                }
            },
        },
    }
}


BATCH_IMAGE_REQUEST_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "images": {"type": "array", "items": {"type": "string", "format": "binary"}},
                    },
                }
            },
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["images"],
                    "properties": {"images": {"type": "array", "items": {"type": "string"}}},
                }
            },
        },
    }
}


@dataclass
class ImageRequest:
    paths: list[Path]
    fields: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.paths[0]

    def cleanup(self) -> None:
        for path in self.paths:
            path.unlink(missing_ok=True)


def _decode_base64(value: str) -> bytes:
    encoded = value.split(",", 1)[1] if "," in value else value
    max_encoded_length = ((_max_upload_bytes() + 2) // 3) * 4 + 4
    if len(encoded) > max_encoded_length:
        raise ModelAPIError(
            413,
            "PAYLOAD_TOO_LARGE",
            "The Base64 image exceeds the configured image size limit.",
            details=[{"max_bytes": _max_upload_bytes()}],
        )
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ModelAPIError(
            422,
            "INVALID_IMAGE_BASE64",
            "The image field is not valid Base64.",
            details=[{"field": "image", "issue": "invalid_base64"}],
        ) from exc


def _write_verified_image(data: bytes) -> Path:
    if not data:
        raise ModelAPIError(422, "EMPTY_IMAGE", "The uploaded image is empty.")
    if len(data) > _max_upload_bytes():
        raise ModelAPIError(
            413,
            "PAYLOAD_TOO_LARGE",
            f"Image exceeds the {limits_settings.upload_mb} MB limit.",
            details=[{"received_bytes": len(data), "max_bytes": _max_upload_bytes()}],
        )
    try:
        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            max_pixels = limits_settings.image_pixels
            max_dimension = limits_settings.image_dimension
            if width <= 0 or height <= 0 or width > max_dimension or height > max_dimension or width * height > max_pixels:
                raise ModelAPIError(
                    413,
                    "IMAGE_DIMENSIONS_TOO_LARGE",
                    "The image dimensions exceed the configured limit.",
                    details=[
                        {
                            "width": width,
                            "height": height,
                            "max_dimension": max_dimension,
                            "max_pixels": max_pixels,
                        }
                    ],
                )
            source.verify()
            image_format = (source.format or "PNG").lower()
    except (UnidentifiedImageError, OSError) as exc:
        raise ModelAPIError(
            422,
            "INVALID_IMAGE",
            "The uploaded content is not a readable image.",
            details=[{"field": "image", "issue": "decode_failed"}],
        ) from exc
    suffix = ".jpg" if image_format in {"jpg", "jpeg"} else f".{image_format}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        return Path(handle.name)


async def _read_upload_limited(upload: Any) -> bytes:
    max_bytes = _max_upload_bytes()
    chunks: list[bytes] = []
    received = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        received += len(chunk)
        if received > max_bytes:
            raise ModelAPIError(
                413,
                "PAYLOAD_TOO_LARGE",
                "An uploaded image exceeds the configured image size limit.",
                details=[{"received_bytes": received, "max_bytes": max_bytes}],
            )
        chunks.append(chunk)
    return b"".join(chunks)


@timed_stage("receive_decode_verify_images")
async def parse_image_request(request: Request, *, multiple: bool = False) -> ImageRequest:
    """Accept image input as multipart upload or JSON Base64/Data URL."""
    content_type = request.headers.get("content-type", "").lower()
    fields: dict[str, Any] = {}
    raw_images: list[bytes] = []

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        for key, value in form.multi_items():
            if hasattr(value, "read"):
                if key in {"image", "images"}:
                    raw_images.append(await _read_upload_limited(value))
                continue
            if key in fields:
                existing = fields[key]
                fields[key] = [existing, value] if not isinstance(existing, list) else [*existing, value]
            else:
                fields[key] = value
        base64_value = fields.pop("image_base64", None)
        if base64_value and not raw_images:
            values = base64_value if isinstance(base64_value, list) else [base64_value]
            raw_images.extend(_decode_base64(str(value)) for value in values)
    elif content_type.startswith("application/json"):
        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ModelAPIError(400, "INVALID_JSON", "The request body is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ModelAPIError(422, "VALIDATION_ERROR", "The JSON body must be an object.")
        values = payload.get("images") if multiple else payload.get("image", payload.get("image_base64"))
        if values is not None:
            image_values = values if isinstance(values, list) else [values]
            raw_images.extend(_decode_base64(str(value)) for value in image_values)
        fields = {key: value for key, value in payload.items() if key not in {"image", "images", "image_base64"}}
    else:
        raise ModelAPIError(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            "Use multipart/form-data or application/json with a Base64 image.",
            details=[{"received": content_type or "missing"}],
        )

    if not raw_images:
        field_name = "images" if multiple else "image"
        raise ModelAPIError(
            422,
            "IMAGE_REQUIRED",
            f"The {field_name} field is required.",
            details=[{"field": field_name, "issue": "required"}],
        )
    if not multiple and len(raw_images) != 1:
        raise ModelAPIError(422, "TOO_MANY_IMAGES", "This endpoint accepts exactly one image.")
    if multiple:
        check_batch_limits(raw_images)
    paths: list[Path] = []
    try:
        for data in raw_images:
            paths.append(_write_verified_image(data))
        return ImageRequest(paths=paths, fields=fields)
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        raise


def parse_bool(value: Any, *, field: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ModelAPIError(
        422,
        "VALIDATION_ERROR",
        f"{field} must be a boolean.",
        details=[{"field": field, "issue": "boolean_required", "received": _safe_received(value)}],
    )


def parse_float(
    value: Any,
    *,
    field: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelAPIError(
            422,
            "VALIDATION_ERROR",
            f"{field} must be numeric.",
            details=[{"field": field, "issue": "number_required", "received": _safe_received(value)}],
        ) from exc
    if (minimum is not None and parsed < minimum) or (maximum is not None and parsed > maximum):
        raise ModelAPIError(
            422,
            "VALIDATION_ERROR",
            f"{field} is outside the allowed range.",
            details=[{"field": field, "minimum": minimum, "maximum": maximum, "received": parsed}],
        )
    return parsed


def parse_json_field(value: Any, *, field: str, default: Any = None) -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ModelAPIError(
            422,
            "VALIDATION_ERROR",
            f"{field} must contain valid JSON.",
            details=[{"field": field, "issue": "invalid_json"}],
        ) from exc

