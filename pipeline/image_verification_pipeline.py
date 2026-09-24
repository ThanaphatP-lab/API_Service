"""Compatibility exports; scoring lives in pipelines.verification."""
from pipelines.verification.scoring import (
    DEFAULT_CATEGORIES, _ui_percentages, normalize_categories,
    verify_classification, verify_classification_targets,
)
