# Phase 5 — topology migration (source ready; runtime gates pending)

## Linux launcher update

At the user's request, `scripts/model-stack.sh start detection` now starts one DET
process on DETECTION_PORT (default 8002); all Linux profiles use it instead of two
version-specific services. The launcher automatically exports TEXT_DETECTION_URL,
so its unified routing is no longer opt-in. The sections below describing opt-in
still apply to services launched manually or by the unchanged Windows launchers.
See deploy/linux/README.md for legacy-process migration. Registry names/directories
are authoritative for detection; old DET_MODEL_NAME/DET_MODEL_DIR overrides must
be migrated into model_variants.json. No VPS services were changed by this edit.
Actual GPU memory/performance validation remains outstanding.

## Implemented

- Optional `TEXT_DETECTION_URL` sends explicitly versioned DET single/batch
  requests directly to one existing `services.text_det.main:app` process.
  That service already selects/caches models by version and variant. No new
  FastAPI service, port or cache policy is introduced.
- When unset/empty, Gateway retains `DET_V5_URL` / `DET_V6_URL` routing.
- Layout and OCR Custom prefer `TEXT_DETECTION_URL` over `DET_SERVICE_URL`
  when configured. Set it consistently in each affected service environment.
- REC already routes directly from Gateway to its leaf service; no extra hop
  was removed from that path in this phase.
- `/api/v1/services` retains its original static route map and adds `capabilities`,
  `capability_status`, `detection_topology` and compatibility notes. Capabilities
  list configured services, not detected running processes or available weights.
  Discovery makes no HTTP/model-loading calls.
- Discovery/readiness use the same enabled/required configuration validation.
  Invalid or empty enabled configuration now also returns
  `503 GATEWAY_CONFIGURATION_ERROR` from discovery.
- Readiness results include `kind` (`leaf` / `pipeline`) and descriptive
  `service_id`. Existing metadata service names, launch commands and directories
  remain unchanged for compatibility.
- OCR Custom recognition-only routes are marked deprecated in OpenAPI and log
  their usage; they remain callable with unchanged response adapters.

## Configuration

Existing environment variables continue to work:

```bash
GATEWAY_ENABLED_PIPELINES=ocr-custom,text-det-v5,text-det-v6,text-recognition
GATEWAY_REQUIRED_PIPELINES=text-recognition
```

New names describe both leaf and pipeline services more accurately:

```bash
GATEWAY_ENABLED_SERVICES=pipeline-ocr-custom,leaf-text-detection-v5,leaf-text-detection-v6,leaf-text-recognition
GATEWAY_REQUIRED_SERVICES=leaf-text-recognition
```

`*_SERVICES` takes precedence over `*_PIPELINES`, including an explicitly empty
value. Use either short legacy names or descriptive service IDs from
`core/topology.py`. The default remains `all` enabled, none required.
These settings control discovery/readiness only; they do not disable HTTP routes.
Set them to the services actually deployed. Optional down services yield degraded
readiness; failed required services or all enabled services down yield 503.

Only after the combined-process resource test passes, opt in:

```bash
TEXT_DETECTION_URL=http://127.0.0.1:8002
GATEWAY_ENABLED_SERVICES=leaf-text-detection,leaf-text-recognition,pipeline-ocr-custom
GATEWAY_REQUIRED_SERVICES=leaf-text-detection,leaf-text-recognition
```

This address is an example using an existing DET port, not a new port allocation.
Restart Gateway and affected pipelines to reload URL snapshots. In unified mode,
legacy `text-det-v5` / `text-det-v6` config aliases collapse to `text-detection`,
so readiness probes the combined service once.

**Readiness limitation:** DET readiness loads the configured default baseline;
it does not certify every version or fine-tuned variant. Successful requests for
both v5/v6 and all intended variants are required independently before cutover.

## Compatibility gates: do not remove yet

1. **Unversioned DET:** Gateway still routes to Layout Pipeline because its legacy
   `regions` / dimensions response differs from the leaf's model output. Merely
   changing the destination would break callers. Versioned DET is direct; migrating
   unversioned callers requires response-contract verification or a compatible adapter.
2. **OCR Custom recognition pass-through:** verify no consumers use it before removal.
   Its normalized response differs from leaf REC output, so switching a caller URL
   alone may not be sufficient. Audit backend parsing and compare responses first.
3. **Physical service consolidation/renaming:** source routing support is not evidence
   that two models fit GPU memory. No running service has been stopped or renamed.

## Linux/GPU acceptance checklist

Use a staging environment with the same GPU, weights and request limits as the VPS.
Run the existing DET service on its existing port with one worker, then:

- Record split-service latency, process RSS and GPU memory using
  `scripts/capture-phase0-baseline.py` and the PHASE0_BASELINE.md procedure.
- Test `/api/v1/text-detections` and `/api/v1/text-detection-batches` directly against
  the candidate process, alternating `version=5` and `version=6`, then all production
  `model` variants. Verify `model_selection`, image count/order and output equivalence.
- Include cold loads, repeated warm calls, mixed-size images, max allowed batches
  and concurrent requests. Keep MAX_CONCURRENT_REQUESTS at the production setting.
- Monitor GPU memory after both models are resident and after variant switching;
  verify headroom for REC/TableV2 and other colocated processes. An LRU entry count
  is not a GPU-memory budget, and multiple workers multiply model residency.
- Record peak memory, OOM/errors and p50/p95 latency against the current baseline.
  Establish acceptable thresholds for the VPS before declaring this gate passed.
- Configure TEXT_DETECTION_URL only after passing; verify Gateway and full OCR/layout
  paths. Stop the redundant DET process only after all callers have migrated and
  the agreed observation window passes.
- Search gateway, backend and OCR Custom access logs for recognition-only calls
  over a representative observation window before scheduling endpoint removal.

Rollback: unset TEXT_DETECTION_URL, restore split DET processes/URLs and the prior
enabled/required service configuration, then restart affected Gateway/pipelines.
Weights and variant configuration are unchanged by this migration.

## Verification performed locally

101 offline tests passed, including unified/split routing, model field forwarding,
single/batch behavior, discovery filtering, config aliases, readiness deduplication,
and existing API contracts. No Linux/GPU resource or production usage result is
claimed. Phase 5 remains partially complete until the gates above pass.
