# Phase 6 — approved cleanup items 1–4 completed

Removed with user approval:
- Old pipeline/ and models/ import wrappers and shared/api.py.
  Tests now import pipelines/, inference/ and core/ directly.
- OCR Custom recognition-only compatibility code and its registration flag.
  Those routes return 404; use Gateway or REC directly.
- DET-only routes and workflow functions from Layout Pipeline.
  document-layouts remains; Gateway calls DET directly with legacy formatting.
- det-v5/det-v6 launcher targets and their PID management on Linux/Windows.
  Use detection and DETECTION_PORT (default 8002).
- Previously removed: models/version_old files and unused local OCR fallback.

No weights, uploads, logs, cache, virtual environments or running processes were
deleted. Tracked source files can be recovered from Git history. The untracked
REC compatibility file introduced during refactoring came from the older committed
OCR Custom main.py, where its original implementation can be recovered.

## Migration

Stop legacy split services using the PREVIOUS launcher or your process manager
before upgrading: the new launcher cannot manage their old PID files.
Move any DET_V5_PORT override to DETECTION_PORT.
Update external Python imports to their new owning modules.
Clients calling REC directly on OCR Custom or DET on Layout must move to Gateway
or leaf APIs and verify response shape. No automatic redirect is provided.
OCR_CUSTOM_RECOGNITION_COMPAT_ENABLED is obsolete and cannot restore removed routes.

## Retained

OCR Custom/PaddleOCR, Table Custom/TableV2 and document-layout workflows remain.
Gateway public paths and DET model-variant selection remain.
LEGACY_ENDPOINTS_ENABLED still controls hidden inference aliases; /health remains.
Model weight directories named det-v5/det-v6 are NOT launcher targets and remain.
Old discovery configuration aliases remain independent of launcher commands.
No tracked generated artifacts were found to delete; existing ignore rules remain.
No GPU inference or VPS rollout was performed.
