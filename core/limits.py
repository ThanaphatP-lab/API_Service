"""Request/image/batch limit policy; parsing stays in request_parsing."""
from shared.contracts import ModelAPIError
from core.settings import limits_settings


def _max_upload_bytes() -> int:
    return limits_settings.upload_bytes


def check_request_size(content_length: str) -> None:
    if not content_length:
        return
    try:
        received_bytes = int(content_length)
    except ValueError as exc:
        raise ModelAPIError(400, "INVALID_CONTENT_LENGTH", "Content-Length must be numeric.") from exc
    max_request_bytes = limits_settings.request_bytes
    if received_bytes > max_request_bytes:
        raise ModelAPIError(
            413, "PAYLOAD_TOO_LARGE",
            "The request body exceeds the configured size limit.",
            details=[{"received_bytes": received_bytes, "max_bytes": max_request_bytes}],
        )


def check_batch_limits(raw_images: list[bytes]) -> None:
    max_images = limits_settings.batch_images
    if len(raw_images) > max_images:
        raise ModelAPIError(
            413,
            "BATCH_TOO_LARGE",
            "The image batch contains too many items.",
            details=[{"received_images": len(raw_images), "max_images": max_images}],
        )
    max_batch_bytes = limits_settings.batch_bytes
    total_bytes = sum(len(data) for data in raw_images)
    if total_bytes > max_batch_bytes:
        raise ModelAPIError(
            413,
            "BATCH_TOO_LARGE",
            "The image batch exceeds the configured total size limit.",
            details=[{"received_bytes": total_bytes, "max_bytes": max_batch_bytes}],
        )
