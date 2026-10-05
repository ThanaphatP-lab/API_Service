# Official OCR variants and request timing

Gateway endpoints (POST):

- `/api/v1/ocr-results?engine=paddle&det_version=6&rec_version=5&rec_model=thai_ft_v1`

Both single and batch support independent `det_version` / `rec_version` (5/v5,
6/v6). Each overrides the common `version` (or `ocr_version`) for that module.
If neither an independent nor common version is supplied, that module uses
PADDLE_OCR_VERSION (default v5) when registry selection is active. Non-baseline
variants at the gateway require a common version or their own module version.
Mixed-generation response metadata contains each module's actual version and
a pair version such as `det-v6__rec-v5`. Cache keys include both selections.

- `/api/v1/ocr-results?engine=paddle&version=5&model=thai_ft_v1`
- `/api/v1/ocr-result-batches?engine=paddle&version=5&det_model=baseline&rec_model=thai_ft_v1`

The pipeline service exposes the same paths without requiring `engine`.
Query parameters override matching multipart/JSON fields. `ocr_version` and
`profile` are aliases for `version` and `model`, as in TableV2.
The shared model_variants.json catalog resolves exported inference weights;
unsupported variants and missing required local weights return catalog errors.
`model` selects both DET and REC; use `rec_model` to change only REC.
Gateway requires a version for non-baseline variants.

Without selection parameters, the legacy DET_MODEL_NAME/DIR and REC_MODEL_NAME/DIR
configuration is preserved, including mixed DET v6 / REC v5 defaults.
Explicit version selection uses the central registry. Direct variant-only calls
default to PADDLE_OCR_VERSION=v5. Set PADDLE_OCR_MODEL_CACHE_SIZE (default 2) before
startup; each cached pipeline may allocate its own DET and REC GPU weights.
Restart after updating model files or registry configuration.

All create_app services emit INFO logs via uvicorn.error:

- request_received / request_complete: request ID, service, status, bytes consumed
  and sent, and elapsed time through ASGI response handling (not client receipt).
- admission_complete: admission semaphore wait, separate from the REC fair queue.
- stage_complete: image receive/decode/verification, inference including cold
  model loading and output adaptation, and upstream HTTP round trips.
- fair_queue_turn: REC wait per turn, chunk size and active jobs.

Resource snapshots include process CPU seconds accumulated since startup and,
on Linux, process peak RSS in MB. They are not request-specific usage, current
RSS, CPU percentage, or GPU utilization. GPU metrics are not collected.
Inference timing is wall-clock pipeline timing, not a CUDA kernel benchmark.
Stages overlap across services: do not sum gateway and leaf durations.
No uploaded image contents, query strings or auth headers are logged by telemetry.
