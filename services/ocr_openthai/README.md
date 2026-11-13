# OpenThai OCR

POST /paddle/api/v1/openthai-text on the Gateway or leaf service.
Existing OCR engine routes are unchanged. Accepts PNG/JPEG in multipart field image (single) or repeated images fields
(batch), or equivalent JSON/base64 input. Response remains {data, meta}.
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
OPENTHAI_REQUEST_TIMEOUT_SECONDS defaults to 120. Batch requests use the same endpoint.

## Authentication
Only INTERNAL_API_TOKEN (and INTERNAL_API_TOKEN_PREVIOUS during rotation) is
accepted by this service through Bearer or X-API-Key. OPENTHAI_API_KEY is ignored.
Missing server token returns 503 AUTH_CONFIGURATION_ERROR for inference/readiness
even in development. A missing/wrong client token returns 401. Health endpoints,
development documentation and OPTIONS retain the shared infrastructure exemptions.
Restart after changing environment tokens. Clients holding an internal token can
call directly if the service port is reachable; authentication does not verify
that a request originated from the Gateway.


## Linux/WSL: existing stack scripts
Install the separate OpenThai environment with Python 3.11/3.12:

    bash scripts/setup-linux.sh openthai

The all setup target includes OpenThai too. Configure .env.runtime using the
existing stack convention (chmod 600). Set INTERNAL_API_TOKEN, MODEL_GATEWAY_API_KEY
when running Gateway, and OPENTHAI_MODEL_DIR to the verified snapshot.
Default OpenThai port is 8091; OPENTHAI_PORT and OPENTHAI_PYTHON can override it.
No OPENTHAI_API_KEY is used.

    bash scripts/model-stack.sh start openthai
    bash scripts/model-stack.sh readiness openthai
    bash scripts/model-stack.sh status openthai
    bash scripts/model-stack.sh logs openthai
    bash scripts/model-stack.sh logs openthai --log-level debug
    bash scripts/model-stack.sh restart openthai
    bash scripts/model-stack.sh stop openthai

To start OpenThai plus Gateway, use start openthai-stack. If Gateway is already
running, restart openthai-stack to apply the profile's service list and URLs.
OpenThai is also included in all, but not core-stack. CPU-only OpenThai targets
do not require an NVIDIA GPU. Logs/PIDs/readiness/auth use the existing manager.


## Batch on the same endpoint
POST /paddle/api/v1/openthai-text:
- Single: multipart image file, or JSON {"image": "<base64>"}.
  Existing data.text/data.lines response is unchanged.
- Batch: repeated multipart images files, or JSON {"images": ["<base64>", "..."]}.
  Returns data.results in input order and data.count. An images field containing
  one image still returns batch format. Repeated image files / JSON image arrays
  are also treated as batch.
- Do not mix image and images in JSON.
Authentication is unchanged. Gateway forwards every image and preserves batch mode.

Postman: Body -> form-data -> add multiple rows named images, Type File.
No engine, batch flag or new URL is required.

Processing is sequential in one model worker, not tensor-level ONNX batching.
MAX_BATCH_IMAGES and MAX_BATCH_UPLOAD_MB apply alongside shared request limits.
OPENTHAI_MAX_UPLOAD_BYTES applies per image. OPENTHAI_REQUEST_TIMEOUT_SECONDS and
OPENTHAI_MAX_RESPONSE_BYTES apply to the entire batch. A failed image fails the
request (no partial-success payload). After timeout the current native call can
finish; the deadline prevents starting subsequent images. Temporary files are
cleaned even when the request fails or times out.
