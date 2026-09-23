#!/usr/bin/env python3
"""Capture API v1 responses, latency, RAM/VRAM, and legacy-route usage.

Run this on the deployed Linux host while the Gateway and its selected
upstreams are healthy.  The script never writes the API token to its output.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


CASES = [
    ("health", "GET", "/api/v1/health", None, None),
    ("readiness", "GET", "/api/v1/readiness", None, None),
    ("services", "GET", "/api/v1/services", None, None),
    ("document-layouts", "POST", "/api/v1/document-layouts", "single", None),
    ("ocr-custom", "POST", "/api/v1/ocr-results?engine=custom", "single", None),
    ("ocr-paddle", "POST", "/api/v1/ocr-results?engine=paddle", "single", None),
    ("ocr-result-batches", "POST", "/api/v1/ocr-result-batches?engine=paddle", "batch", None),
    ("text-detections-v5", "POST", "/api/v1/text-detections?version=5", "single", None),
    ("text-detection-batches-v5", "POST", "/api/v1/text-detection-batches?version=5", "batch", None),
    ("text-recognitions-v5", "POST", "/api/v1/text-recognitions?version=5", "single", None),
    ("text-recognition-batches-v5", "POST", "/api/v1/text-recognition-batches?version=5", "batch", None),
    ("table-results", "POST", "/api/v1/table-results", "single", None),
    ("table-model-results", "POST", "/api/v1/table-model-results?ocr_version=5&profile=baseline", "single", None),
    ("image-classifications", "POST", "/api/v1/image-classifications", "single", {"labels": '["a document"]'}),
    ("image-verifications", "POST", "/api/v1/image-verifications", "single", {"image_category": "qr_code"}),
]

LEGACY_PATTERNS = {
    "predict": re.compile(r'\"(?:POST|GET) /predict(?:\?| )'),
    "v1_textdetection": re.compile(r'\"POST /v1/textdetection(?:\?| )'),
    "v1_textrecognition": re.compile(r'\"POST /v1/textrecognition(?:\?| )'),
    "legacy_health": re.compile(r'\"GET /health(?:\?| )'),
}


def _command(args: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _legacy_usage(log_dir: Path) -> dict[str, Any]:
    counts = {name: 0 for name in LEGACY_PATTERNS}
    scanned: list[str] = []
    if not log_dir.is_dir():
        return {"log_dir": str(log_dir), "files": scanned, "counts": counts}
    for path in sorted(log_dir.glob("*.log")):
        scanned.append(path.name)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name, pattern in LEGACY_PATTERNS.items():
            counts[name] += len(pattern.findall(content))
    return {"log_dir": str(log_dir), "files": scanned, "counts": counts}


def _request_case(
    session: requests.Session,
    *,
    base_url: str,
    name: str,
    method: str,
    path: str,
    upload_mode: str | None,
    fields: dict[str, str] | None,
    image_bytes: bytes,
    image_name: str,
    timeout: float,
) -> dict[str, Any]:
    files = None
    if upload_mode == "single":
        files = {"image": (image_name, image_bytes, "application/octet-stream")}
    elif upload_mode == "batch":
        files = [
            ("images", (image_name, image_bytes, "application/octet-stream")),
            ("images", (image_name, image_bytes, "application/octet-stream")),
        ]
    started = time.perf_counter()
    try:
        response = session.request(
            method,
            f"{base_url.rstrip('/')}{path}",
            files=files,
            data=fields,
            timeout=timeout,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        try:
            body: Any = response.json()
        except ValueError:
            body = {"non_json_body": response.text[:2000]}
        return {
            "name": name,
            "method": method,
            "path": path,
            "status": response.status_code,
            "elapsed_ms": elapsed_ms,
            "request_id": response.headers.get("X-Request-ID"),
            "body": body,
        }
    except requests.RequestException as exc:
        return {
            "name": name,
            "method": method,
            "path": path,
            "status": None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default=os.getenv("MODEL_GATEWAY_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--token", default=os.getenv("MODEL_GATEWAY_API_KEY", ""))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()

    image_path = args.image.resolve()
    if not image_path.is_file():
        parser.error(f"sample image does not exist: {image_path}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or Path("baselines") / f"phase0-{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    if args.token:
        session.headers["Authorization"] = f"Bearer {args.token}"
    image_bytes = image_path.read_bytes()
    responses = [
        _request_case(
            session,
            base_url=args.gateway,
            name=name,
            method=method,
            path=path,
            upload_mode=upload_mode,
            fields=fields,
            image_bytes=image_bytes,
            image_name=image_path.name,
            timeout=args.timeout,
        )
        for name, method, path, upload_mode, fields in CASES
    ]
    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "gateway": args.gateway,
        "sample_image": image_path.name,
        "responses": responses,
        "system": {
            "process_memory": _command(["ps", "-eo", "pid,rss,args"]),
            "gpu_memory": _command(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            ),
        },
        "legacy_endpoint_usage": _legacy_usage(args.logs.resolve()),
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = sum(item.get("status") is None or int(item.get("status") or 500) >= 400 for item in responses)
    print(f"wrote {output_path.resolve()}")
    print(f"captured={len(responses)} failures={failures}")
    return 1 if failures and not args.allow_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
