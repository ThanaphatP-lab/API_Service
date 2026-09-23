# Leaf inference contract

Leaf services perform model inference and JSON serialization only. ROI
orchestration, quality gates, threshold decisions, recovery, ranking, and
business logic remain in pipeline services or the Backend.

Every leaf response keeps the platform envelope:

```json
{
  "data": {
    "contract_version": "leaf-inference-v1",
    "kind": "text_recognition",
    "result": {"rec_text": "example", "rec_score": 0.98},
    "raw_output": [],
    "predictions": [{"rec_text": "example", "rec_score": 0.98}]
  },
  "meta": {"api_version": "v1", "service": "...", "model": "..."}
}
```

`result` is the canonical notebook-compatible value. `raw_output` is a strict
JSON-safe copy useful for diagnostics. `predictions` is a compatibility alias
for existing composition pipelines and may be removed only in a future API
version.

Canonical result shapes:

- Layout: `{"detections": [{"bbox", "label", "score", "cls_id", "polygon_points"}]}`
- Text detection: `{"dt_polys": [], "dt_scores": []}`
- Text recognition: `{"rec_text": "", "rec_score": null}`
- Recognition batch: `{"results": [...]}`
- SigLIP: `{"logits": [], "categories": []}`

SigLIP accepts either the existing prompt-list input:

```json
{"labels": ["a QR code", "a document"]}
```

or notebook (4) category objects. Only entries with `enabled: true` are sent
to the model; threshold values remain pipeline/Backend concerns:

```json
{
  "categories": [
    {
      "value": "qr_code",
      "label": "QR Code",
      "prompt": "a QR code",
      "match_threshold": 0.55,
      "margin_threshold": 0.05,
      "evidence_temperature": 1.0,
      "enabled": true
    }
  ]
}
```

## Related: Table V2 integrated pipeline deployment

`TableRecognitionPipelineV2` owns several internal submodels. Its notebook (4)
configuration uses `SLANeXt_wired`, `SLANeXt_wireless`, and
`th_PP-OCRv5_mobile_rec`; document orientation, unwarping, and internal layout
detection are disabled while OCR remains enabled. Run it as an
alternative to the split wired/wireless table stack to avoid duplicate GPU
weights:

```bash
./scripts/model-stack.sh start table-v2-stack
```

Then call the Gateway at `POST /api/v1/table-model-results`. Select the internal
OCR model pair with `ocr_version`/`profile` (or the backward-compatible
`version`/`model` aliases):

```text
POST /api/v1/table-model-results?ocr_version=5&profile=baseline
POST /api/v1/table-model-results?ocr_version=5&profile=thai_ft_v1
POST /api/v1/table-model-results?ocr_version=5&det_model=baseline&rec_model=thai_ft_v1
```

The older `POST /api/v1/table-results` remains the split table composition
pipeline.

With Compose, start only the optional model and Gateway:

```bash
docker compose --profile table-v2 up -d table-v2 gateway
```
