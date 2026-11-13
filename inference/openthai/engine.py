import hashlib
import importlib
import io
import json
import sys
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from shared.contracts import ModelAPIError

def _failure(code, message, status_code=422):
    return ModelAPIError(status_code, code, message)



def decode_image(content, settings):
    if not content:
        raise _failure("IMAGE_REQUIRED", "Upload a nonempty image.")
    if len(content) > settings.max_upload_bytes:
        raise _failure("PAYLOAD_TOO_LARGE", "Image exceeds the upload limit.", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as source:
                if source.format not in {"PNG", "JPEG"}:
                    raise _failure("UNSUPPORTED_MEDIA_TYPE", "Use PNG or JPEG.", 415)
                width, height = source.size
                if width * height > settings.max_pixels or max(width, height) > settings.max_dimension:
                    raise _failure("IMAGE_TOO_LARGE", "Image exceeds the pixel or dimension limit.", 413)
                source.load()
                return source.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise _failure("IMAGE_TOO_LARGE", "Image exceeds safe decoder limits.", 413) from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise _failure("INVALID_IMAGE", "The uploaded image is invalid.") from None


class Inference:
    def __init__(self, settings, loader=None):
        self.settings = settings
        self.loader = loader or self.load
        self.reader = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openthai")
        self.pending = None

    def load(self):
        directory = self.settings.model_dir.resolve()
        manifest = json.loads((Path(__file__).with_name("model-manifest.json")).read_text())
        for name, expected in manifest["sha256"].items():
            with (directory / name).open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected:
                raise ValueError("Model snapshot checksum mismatch")
        # Only the operator-provisioned, revision-verified model directory is imported.
        sys.path.insert(0, str(directory))
        module = importlib.import_module("runtime.line_engine")
        return module.LineOCR(directory, threads=self.settings.threads, runtime={"backend": "onnxruntime"})

    def initialize(self):
        self.reader = self.loader()

    def read(self, content):
        with decode_image(content, self.settings) as image:
            result = self.reader.read(image)
            from inference.openthai.schemas import convert
            return convert(result, image.width, image.height)


    def read_batch(self, contents):
        """Process images in input order as one executor job; no concurrent reader calls."""
        deadline = time.monotonic() + self.settings.request_timeout_seconds
        results = []
        for content in contents:
            if time.monotonic() >= deadline:
                raise _failure("REQUEST_TIMEOUT", "OpenThai processing deadline exceeded.", 504)
            results.append(self.read(content))
        return {"results": results, "count": len(results)}
