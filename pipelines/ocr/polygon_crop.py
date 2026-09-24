from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def _crop_quad(image: np.ndarray, points: Any) -> np.ndarray | None:
    quad = np.asarray(points, dtype=np.float32).copy()
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return None

    area = 0.0
    for index in range(-1, 3):
        area += -0.5 * (quad[index + 1][1] + quad[index][1]) * (quad[index + 1][0] - quad[index][0])
    if area < 0:
        quad[[1, 3]] = quad[[3, 1]]

    width = int(max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3])))
    height = int(max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2])))
    if width <= 0 or height <= 0:
        return None

    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    crop = cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(quad, target),
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.size == 0:
        return None
    crop_height, crop_width = crop.shape[:2]
    if crop_height / float(max(crop_width, 1)) >= 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)
