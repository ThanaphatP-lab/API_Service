import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Line(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = Field(max_length=32768)
    confidence: float | None = Field(default=None, ge=0, le=1)
    row: int = Field(ge=0)
    box: list[tuple[float, float]] = Field(min_length=4, max_length=4)


class OCRData(BaseModel):
    text: str = Field(max_length=32768)
    lines: list[Line] = Field(max_length=1024)
    confidence: None = None
    geometry_available: bool
    image: dict[str, int]
    rotation: Literal[0, 90, 180, 270] = 0
    text_angle: float = 0


def native(value):
    """Accept numeric NumPy scalars/arrays without serializing arbitrary objects."""
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items() if key in {
            "text", "lines", "confidence", "row", "box", "rotation", "text_angle"}}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if hasattr(value, "tolist"):
        return native(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite model output")
    return value


def convert(result, width, height):
    if not isinstance(result, dict) or not isinstance(result.get("lines"), list) or len(result["lines"]) > 1024:
        raise ValueError("Invalid or excessive model lines")
    if not isinstance(result.get("text"), str) or len(result["text"]) > 32768:
        raise ValueError("Invalid or excessive model text")
    if sum(len(line.get("text", "")) for line in result["lines"] if isinstance(line, dict)) > 65536:
        raise ValueError("Excessive model line text")
    result = native(result)
    lines = [Line.model_validate(line) for line in result["lines"]]
    return OCRData(text=result["text"], lines=lines, geometry_available=bool(lines),
                   image={"width": width, "height": height},
                   rotation=result.get("rotation", 0), text_angle=result.get("text_angle", 0)).model_dump()
