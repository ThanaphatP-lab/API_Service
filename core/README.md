# Core infrastructure (Phase 4)

| Module | Responsibility |
| --- | --- |
| `cache.py` | Single-flight LRU model-loader cache, without FastAPI imports |
| `request_parsing.py` | Multipart/Base64 input, image verification, temporary files and field parsing |
| `limits.py` | Request Content-Length and batch count/byte policies |
| `auth.py` | Token configuration/rotation, validation, matching and security headers |
| `app_factory.py` | FastAPI creation, middleware order, concurrency slots, error handlers, health and OpenAPI |
| `readiness.py` | Leaf model readiness route registration |
| `errors.py` | Existing error envelopes, redaction and inference error translation |
| `settings.py` | Typed runtime, upload/batch, rate-limit and layout settings |
| `service_settings.py` | Typed service URL snapshots using existing environment names/defaults |

Import from the owning core module. The shared/api.py compatibility facade was
removed in Phase 6. Tests patch the owning core module directly.

## Preserved behavior

- API routes, request/response schemas, auth exemptions, token rotation and
  production token validation remain unchanged.
- Middleware ordering stays: request ID, legacy route gate, rate limit, auth,
  Content-Length check, inference queue, handler and response headers.
- Default batch maximum remains 64; exceeding it returns 413, not automatic
  splitting. OCR workflow chunking is a separate caller-side responsibility.
- `MAX_REQUEST_MB` is still the existing Content-Length check, not a new ASGI
  streamed-body limiter. Image and batch limits are checked by parsing as before.
- Model cache sizes, GPU initialization and model variant selection are unchanged.
- Service URLs use exactly the existing environment names and port defaults.
  This phase does not consolidate services or alter deployment topology.

## Settings lifecycle

Runtime/limit/layout properties read the environment on access, preserving
request-time behavior. Concurrency and queue timeout are captured by create_app;
RateLimitSettings is captured by each limiter; ServiceURLs is captured when a
service module initializes. Configured inbound tokens are captured at app creation.
As before, deployment changes should restart the affected service.

Integer fallback/clamping and strict conversion behavior are intentionally retained.
No new config library or dependency was added. Model-specific names/weights and
variant resolution remain in the inference/shared model configuration modules;
they were not folded into a generic infrastructure settings object.

## Intentional bug fix

Batch parsing now appends each verified temporary image path immediately.
Previously a list comprehension could fail on a later invalid image before
assigning the paths list, leaving earlier files behind. Error status/schema is
unchanged; the earlier files are now cleaned up. Regression tests cover this.

## Validation

Run `python -m pytest -q tests` in the project development environment.
Tests cover old import compatibility, cache concurrency, API contracts, auth,
queue saturation/release, readiness, request/batch limits, settings lifecycle,
and temporary-file cleanup. Linux/GPU output and performance validation still
requires the deployment baseline procedure in `PHASE0_BASELINE.md`.
