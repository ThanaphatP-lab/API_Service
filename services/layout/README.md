# Layout and SigLIP batch APIs

POST through the Gateway, using repeated multipart `images` fields or JSON
`{"images": ["<base64 image 1>", "<base64 image 2>"]}`:

- `/api/v1/layout-prediction-batches`: Layout leaf results per image.
- `/api/v1/image-classification-batches`: SigLIP results per image; supply a
  shared `categories` or `labels` list (JSON string in multipart, array in JSON).
- `/api/v1/document-layout-batches`: full Layout + DET pipeline per page;
  existing ROI options are shared across the batch. Timing remains per page.

Each endpoint is also available on its respective existing service. Responses
use the normal envelope with `data.results` in input order and `data.count`.
Errors fail the whole request, rather than returning silent partial results.
Single-image routes and their response shapes are unchanged.

`MAX_BATCH_IMAGES` and existing upload limits apply at gateway and service.
`LAYOUT_BATCH_SIZE=1` is the conservative inference default; increase only after
testing model/input-shape compatibility and VRAM on the deployment runtime.
`SIGLIP_BATCH_SIZE=4` bounds actual multi-image model calls, reusing model cache.
Document pipeline batches process pages sequentially with the existing pipeline;
they do not batch downstream calls or introduce GPU parallelism/fair scheduling.
No new ports or services are required. Restart the gateway, layout leaf, SigLIP
and layout pipeline after deploying. Gateway discovery now includes the
`layout-model` leaf; deployments using all-service readiness will probe it too.
