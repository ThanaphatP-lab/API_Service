# OpenThai OCR

POST /paddle/api/v1/openthai-text on the Gateway or leaf service.
Existing OCR engine routes are unchanged. Accepts one PNG/JPEG in multipart
field image, or standard JSON/base64 input. Response remains {data, meta}.
Data preserves OpenThai text, lines (text/confidence/row/box), confidence (null),
geometry_available, image, rotation and text_angle. Direct service metadata
includes the revision in model_version; Gateway metadata describes the Gateway.

## Local launch from models/
Use a separate Python 3.11/3.12 environment:

    python -m pip install -r requirements/openthai.txt
    python -m uvicorn services.ocr_openthai.main:app --host 127.0.0.1 --port 8091 --workers 1

Set OPENTHAI_MODEL_DIR to the model snapshot location before launch.
Set OCR_OPENTHAI_URL=http://localhost:8091 in Gateway and restart it.
Both services share INTERNAL_API_TOKEN; clients use MODEL_GATEWAY_API_KEY.
Add openthai to GATEWAY_ENABLED_SERVICES (or GATEWAY_ENABLED_PIPELINES) for
discovery/readiness, and to the required list only if it should gate readiness.

    curl.exe -X POST http://localhost:8080/paddle/api/v1/openthai-text -H "Authorization: Bearer YOUR_GATEWAY_KEY" -F "image=@sample.png"

## Weights and Docker
Default weights: weights/openthai2/. OPENTHAI_MODEL_DIR overrides this location.
Provision all files in inference/openthai/model-manifest.json, retaining runtime/.
The loader verifies SHA-256 before importing code; it does not download weights.
A mismatched snapshot fails readiness. The existing test snapshot has a different
README checksum from the original API manifest; provision a matching verified
snapshot rather than bypassing verification.

    docker compose --profile openthai up -d --build openthai gateway

The service uses a separate CPU ONNX image. Gateway is configured with
OCR_OPENTHAI_URL=http://openthai:8000. Enable openthai in the gateway service list
as described above when using this optional Compose profile.

## Lifecycle
Health: /paddle/api/v1/health; readiness: /paddle/api/v1/readiness.
Model loads once at startup. Failed loading leaves readiness at 503.
Run exactly one worker. A timed-out native job retains its slot; subsequent
requests receive SERVICE_BUSY until completion.
Shared limits and OPENTHAI_MAX_UPLOAD_BYTES, OPENTHAI_MAX_PIXELS,
OPENTHAI_MAX_DIMENSION, OPENTHAI_MAX_RESPONSE_BYTES apply.
OPENTHAI_REQUEST_TIMEOUT_SECONDS defaults to 120. No batch endpoint is provided.

## Authentication
Only INTERNAL_API_TOKEN (and INTERNAL_API_TOKEN_PREVIOUS during rotation) is
accepted by this service through Bearer or X-API-Key. OPENTHAI_API_KEY is ignored.
Missing server token returns 503 AUTH_CONFIGURATION_ERROR for inference/readiness
even in development. A missing/wrong client token returns 401. Health endpoints,
development documentation and OPTIONS retain the shared infrastructure exemptions.
Restart after changing environment tokens. Clients holding an internal token can
call directly if the service port is reachable; authentication does not verify
that a request originated from the Gateway.
