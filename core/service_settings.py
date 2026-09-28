"""Service addresses. Defaults and environment names preserve existing topology."""
from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class ServiceURLs:
    # Unified DET is the default for manual launches as well as stack scripts.
    text_detection_url: str = field(default_factory=lambda: os.getenv("TEXT_DETECTION_URL", "http://localhost:8002").strip() or "http://localhost:8002")
    layout_pipeline_url: str = field(default_factory=lambda: os.getenv("LAYOUT_PIPELINE_URL", "http://localhost:8010"))
    rec_service_url: str = field(default_factory=lambda: os.getenv("REC_SERVICE_URL", "http://localhost:8004"))
    ocr_custom_url: str = field(default_factory=lambda: os.getenv("OCR_CUSTOM_URL", "http://localhost:8005"))
    ocr_paddle_url: str = field(default_factory=lambda: os.getenv("OCR_PADDLE_URL", "http://localhost:8006"))
    table_pipeline_url: str = field(default_factory=lambda: os.getenv("TABLE_PIPELINE_URL", "http://localhost:8011"))
    table_model_url: str = field(default_factory=lambda: os.getenv("TABLE_MODEL_URL", "http://localhost:8013"))
    siglip_url: str = field(default_factory=lambda: os.getenv("SIGLIP_URL", "http://localhost:8009"))
    image_verification_url: str = field(default_factory=lambda: os.getenv("IMAGE_VERIFICATION_URL", "http://localhost:8012"))
    siglip_service_url: str = field(default_factory=lambda: os.getenv("SIGLIP_SERVICE_URL", "http://localhost:8009"))
    layout_service_url: str = field(default_factory=lambda: os.getenv("LAYOUT_SERVICE_URL", "http://localhost:8001"))
    det_service_url: str = field(default_factory=lambda: os.getenv("DET_SERVICE_URL", "http://localhost:8002"))
    table_wired_service_url: str = field(default_factory=lambda: os.getenv("TABLE_WIRED_SERVICE_URL", "http://localhost:8007"))
    table_wireless_service_url: str = field(default_factory=lambda: os.getenv("TABLE_WIRELESS_SERVICE_URL", "http://localhost:8008"))
    ocr_service_url: str = field(default_factory=lambda: os.getenv("OCR_SERVICE_URL", "http://localhost:8005"))
