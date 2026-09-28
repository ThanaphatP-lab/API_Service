# Pipeline workflow boundaries (Phase 3)

Request flow: `services/*/main.py` -> `pipelines/*/orchestrator.py`
-> injected `clients.ModelClient` -> leaf HTTP service -> `inference/`.

The actual client import is `clients.model_service_client.ModelClient`.
HTTPModelClient delegates to shared.upstream, preserving authentication,
timeouts, request IDs and error translation. Tests inject a fake client.

| Package | Responsibilities |
| --- | --- |
| `ocr` | Orchestration, polygon perspective crop, legacy response contracts |
| `layout` | Remote workflow and separate region geometry/filtering |
| `table` | Grid analysis, HTML parser, OCR cell assignment, quality gates, candidate selection, result assembly and workflow |
| `verification` | One classification call, scoring and target selection |

Service routes own request parsing, response envelopes and uploaded-file cleanup.
Workflows receive URLs, client and request ID as arguments; they do not read
global service URLs. OCR and table workflows own their temporary crop cleanup.
Pure geometry/scoring modules do not make HTTP calls.

## Compatibility and intended behavior

- Old `pipeline/*.py` compatibility exports were removed in Phase 6; use `pipelines/`.
- API paths, response fields, selection fields and quality thresholds are unchanged.
- OCR preserves polygon cropping, recognition chunking and output order.
- Table auto mode preserves initial grid-based selection, shared OCR, semi-region
  batch recovery and alternate-model fallback.
- This custom table workflow is distinct from official TableV2. Official
  TableV2/PaddleOCR initialization and variant capabilities are not extended here.
- Shared API infrastructure lives in `core/`; `shared/api.py` was removed in Phase 6.
  Upstream transport still uses `shared/upstream.py`.

For workflow tests, inject a ModelClient instead of patching transport globals
in the old compatibility modules. No model/GPU is needed for these tests.
Real model output equivalence, latency and GPU memory still require the Linux/GPU
baseline procedure in PHASE0_BASELINE.md before production rollout.
