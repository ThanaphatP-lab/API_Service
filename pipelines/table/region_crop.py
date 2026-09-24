from __future__ import annotations

from pathlib import Path
import tempfile
import cv2


def _semi_region_crops(image_path: Path, regions: list[dict]) -> list[Path]:
    source = cv2.imread(str(image_path))
    if source is None:
        return []
    height, width = source.shape[:2]
    paths: list[Path] = []
    for region in regions:
        x = max(0, min(width - 1, int(region.get("x") or 0)))
        y = max(0, min(height - 1, int(region.get("y") or 0)))
        right = min(width, x + max(1, int(region.get("width") or 1)))
        bottom = min(height, y + max(1, int(region.get("height") or 1)))
        crop = source[y:bottom, x:right]
        if crop.size == 0:
            continue
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        handle.close()
        path = Path(handle.name)
        if cv2.imwrite(str(path), crop):
            paths.append(path)
        else:
            path.unlink(missing_ok=True)
    return paths
