"""Typed infrastructure settings using the existing environment names/defaults.

Properties intentionally read on access: request-time limits and runtime flags
remain dynamic. App construction and service URL constants still snapshot their
values at the same lifecycle stage as before this refactor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _is_production() -> bool:
    return os.getenv("APP_ENV", "development").strip().lower() in {"prod", "production"}


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True)
class RuntimeSettings:
    @property
    def recognition_batch_size(self) -> int:
        return min(_positive_int_env("OCR_RECOGNITION_BATCH_SIZE", 64), limits_settings.batch_images)

    @property
    def gateway_readiness_timeout(self) -> float:
        try:
            return max(0.1, float(os.getenv("GATEWAY_READINESS_TIMEOUT_SECONDS", "10")))
        except ValueError:
            return 10.0

    @property
    def error_docs_url(self) -> str:
        return os.getenv("ERROR_DOCS_URL", "").strip()

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in os.getenv("CORS_ORIGINS", "").split(",") if item.strip()]

    @property
    def queue_timeout(self) -> float:
        return float(os.getenv("INFERENCE_QUEUE_TIMEOUT_SECONDS", "5"))

    def max_concurrent_requests(self, *, public_api: bool) -> int:
        return _positive_int_env("MAX_CONCURRENT_REQUESTS", 2 if public_api else 1)

    @property
    def debug_errors(self) -> bool:
        return _env_flag("DEBUG_ERRORS")

    @property
    def model_device(self) -> str:
        return os.getenv("MODEL_DEVICE", "gpu:0")

    @property
    def trust_proxy_headers(self) -> bool:
        return _env_flag("TRUST_PROXY_HEADERS")


@dataclass(frozen=True)
class LimitsSettings:
    @property
    def upload_mb(self) -> int:
        # Preserve the original strict conversion (invalid values raise).
        return int(os.getenv("MAX_UPLOAD_MB", "20"))

    @property
    def upload_bytes(self) -> int:
        return max(1, self.upload_mb) * 1024 * 1024

    @property
    def request_bytes(self) -> int:
        return _positive_int_env("MAX_REQUEST_MB", 28) * 1024 * 1024

    @property
    def image_pixels(self) -> int:
        return _positive_int_env("MAX_IMAGE_PIXELS", 40_000_000)

    @property
    def image_dimension(self) -> int:
        return _positive_int_env("MAX_IMAGE_DIMENSION", 12_000)

    @property
    def batch_images(self) -> int:
        return _positive_int_env("MAX_BATCH_IMAGES", 64)

    @property
    def batch_bytes(self) -> int:
        return _positive_int_env("MAX_BATCH_UPLOAD_MB", 64) * 1024 * 1024


@dataclass(frozen=True)
class RateLimitSettings:
    requests: int = field(default_factory=lambda: max(0, int(os.getenv("RATE_LIMIT_REQUESTS", "120"))))
    window_seconds: int = field(default_factory=lambda: max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))))
    max_keys: int = field(default_factory=lambda: max(100, int(os.getenv("MAX_RATE_LIMIT_KEYS", "10000"))))


@dataclass(frozen=True)
class LayoutSettings:
    @property
    def padding(self) -> tuple[int, int, int, int]:
        return (
            int(os.getenv("AUTO_ROI_EXPAND_TOP_PX", "8")),
            int(os.getenv("AUTO_ROI_EXPAND_RIGHT_PX", "8")),
            int(os.getenv("AUTO_ROI_EXPAND_BOTTOM_PX", "8")),
            int(os.getenv("AUTO_ROI_EXPAND_LEFT_PX", "8")),
        )

    @property
    def table_padding(self) -> tuple[int, int, int, int]:
        return (
            int(os.getenv("AUTO_ROI_TABLE_EXPAND_TOP_PX", "2")),
            int(os.getenv("AUTO_ROI_TABLE_EXPAND_RIGHT_PX", "2")),
            int(os.getenv("AUTO_ROI_TABLE_EXPAND_BOTTOM_PX", "2")),
            int(os.getenv("AUTO_ROI_TABLE_EXPAND_LEFT_PX", "2")),
        )

    @property
    def max_neighbor_overlap(self) -> float:
        return float(os.getenv("AUTO_ROI_MAX_NEIGHBOR_OVERLAP_RATIO", "0.15"))


runtime_settings = RuntimeSettings()
limits_settings = LimitsSettings()
layout_settings = LayoutSettings()
