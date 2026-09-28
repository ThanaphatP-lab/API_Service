# Phase 5 — unified service topology

## Completion and evidence

DET resource acceptance was explicitly confirmed by the user on 2026-09-25.
No new GPU benchmark is claimed by this refactor. Source changes and offline
regressions verify routing/contracts; live VPS rollout remains an operator action.

- DET v5/v6 use one service and cache models by version/variant like REC.
- Linux and Windows stack profiles start `detection` once (default port 8002).
- Manual Gateway launches also default to TEXT_DETECTION_URL=http://localhost:8002.
  DET_V5_URL/DET_V6_URL no longer choose separate Gateway destinations.
- Both versioned and unversioned DET go directly to the leaf. For unversioned
  requests Gateway requests the existing legacy response formatter inside the leaf;
  regions, dimensions and raw_predictions remain compatible without a Layout hop.
- REC Gateway routes go directly to REC as before.
- Service metadata uses leaf-* / pipeline-* identifiers matching its responsibility.
  Consumers matching literal meta.service strings must update their mappings.
- Discovery/readiness use configured enabled/required services and recognize old
  text-det-v5/text-det-v6 config aliases as one text-detection dependency.
- OCR Custom recognition-only endpoints were permanently removed in Phase 6.
  Use Gateway/REC directly; the old compatibility flag has no effect.

## Start and migrate

Stop old split services with the previous launcher/process manager BEFORE upgrading.
Linux (after old processes have stopped):

```bash
./scripts/model-stack.sh start detection
./scripts/model-stack.sh restart gateway
# Restart only deployed pipelines:
./scripts/model-stack.sh restart ocr-custom
./scripts/model-stack.sh restart layout-pipeline
```

Windows uses `scripts\model-stack.cmd start detection` (same service semantics).
Old det-v5/det-v6 targets were removed in Phase 6, including stop/status/logs.
The new launcher does not manage old PID files.

For manually launched services set TEXT_DETECTION_URL to the DET address.
The stack scripts export it automatically from DETECTION_PORT (default 8002). Use one worker to avoid duplicate GPU caches.

Names and weights are selected from the detection section in model_variants.json.
Linux/Windows detection launchers clear DET_MODEL_NAME/DET_MODEL_DIR overrides;
move old overrides into the registry. MODEL_VARIANTS_CONFIG selects an alternative
registry, DET_MODEL_VERSION chooses the default version, and model defaults to
baseline. Restart after changing registry configuration.

```text
POST /api/v1/text-detections?version=5
POST /api/v1/text-detections?version=6&model=thai_ft_v1
POST /api/v1/text-detection-batches?version=6&model=thai_ft_v1
```

Deploy updated DET and Gateway together: the direct unversioned path requires
the leaf's new legacy formatter. Test single/batch and both model versions after
restart. Readiness loads only the default baseline, not every configured variant.

## Discovery and readiness

GATEWAY_ENABLED_SERVICES / GATEWAY_REQUIRED_SERVICES override their older
GATEWAY_ENABLED_PIPELINES / GATEWAY_REQUIRED_PIPELINES counterparts, including
explicit empty values. Use legacy short names or descriptive service IDs in
core/topology.py. Example:

```bash
GATEWAY_ENABLED_SERVICES=leaf-text-detection,leaf-text-recognition,pipeline-ocr-custom
GATEWAY_REQUIRED_SERVICES=leaf-text-detection,leaf-text-recognition
```

Discovery lists configured capabilities, not live process or weight availability.
Readiness probes only configured services. These settings do not disable routes.
The original static route map remains available for older discovery consumers.

## Rollback and limitations

Restoring removed compatibility requires a previous code release, not an environment flag.

Full split-topology rollback requires the previous release of scripts/Gateway,
previous config and restarting the former processes. Do not unset the unified
URL expecting the current Gateway to revert to two ports.

Workspace backend references were audited: model_runtime_client.py calls DET/REC
through Gateway. External/VPS clients must migrate their imports/routes before adopting Phase 6. No weights, deployed processes or Docker settings were changed.
