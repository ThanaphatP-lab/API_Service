# Model Platform Refactoring Plan

เอกสารนี้เป็นแผนปรับโครงสร้างโค้ดในโฟลเดอร์ `models/` ให้แยกความรับผิดชอบระหว่าง
Gateway, HTTP service, pipeline orchestration และ model inference อย่างชัดเจน โดยต้อง
รักษา public API เดิมระหว่างการย้ายระบบ และต้องสามารถเพิ่ม model version/variant ใหม่ได้
โดยไม่สร้าง endpoint หรือ service ซ้ำโดยไม่จำเป็น

## 1. เป้าหมาย

- ทำให้เส้นทาง `Gateway -> Pipeline/Leaf -> Model` อ่านและตรวจสอบได้ง่าย
- แยก HTTP transport ออกจาก workflow และ model inference
- ใช้ model catalog เป็นแหล่งข้อมูลกลางสำหรับชื่อโมเดลและตำแหน่ง weight
- รองรับ baseline และ fine-tuned variants โดยไม่โหลดโมเดลใหม่ทุก request
- ลดโค้ดซ้ำระหว่าง DET, REC, OCR และ TableV2
- รักษา `/api/v1` และ default `baseline` ให้ backward compatible
- ทำให้ unit test รันได้โดยไม่จำเป็นต้องโหลด Paddle/GPU ทุกกรณี
- ลด configuration drift ระหว่าง source, Linux scripts, Windows scripts และ Compose

## 2. สิ่งที่ไม่ทำในรอบเดียว

- ไม่เปลี่ยน public API ทั้งหมดพร้อมกัน
- ไม่ลบ legacy endpoint ก่อนตรวจ access log และประกาศ deprecation
- ไม่รวมทุก service เป็น process เดียว
- ไม่ย้าย business logic ไปไว้ใน Gateway
- ไม่เปลี่ยน model output contract โดยไม่มี contract test
- ไม่ลบ weight หรือ model artifact อัตโนมัติ

## 3. สถาปัตยกรรมปัจจุบัน

```text
Backend/Client
    |
    v
Gateway :8080
    |-- Layout pipeline :8010
    |     |-- Layout leaf :8001
    |     `-- Detection leaf :8002/:8003
    |
    |-- Custom OCR pipeline :8005
    |     |-- Detection leaf
    |     `-- Recognition leaf :8004
    |
    |-- PaddleOCR integrated pipeline :8006
    |
    |-- Legacy table pipeline :8011
    |     |-- Wired table leaf :8007
    |     |-- Wireless table leaf :8008
    |     `-- Custom OCR pipeline
    |
    |-- TableRecognitionPipelineV2 :8013
    |
    `-- Image verification pipeline :8012
          `-- SigLIP leaf :8009
```

### ปัญหาหลักของโครงสร้างปัจจุบัน

1. Model loader บางตัวอยู่ใน `models/` แต่บางตัวอยู่ใน `services/` โดยตรง
2. Pipeline บางไฟล์รวมทั้ง HTTP call, image operation, compatibility contract และ workflow
3. `shared/api.py` รับผิดชอบหลายเรื่องเกินไป ทั้ง auth, middleware, parsing, limits,
   readiness, cache และ error handling
4. ค่า URL, port และ weight path ซ้ำในหลายไฟล์ ทำให้เกิด configuration drift
5. ชื่อ `version` ถูกใช้ได้หลายความหมาย เช่น API version, pipeline generation และ OCR generation
6. Legacy route และ pass-through route ทำให้ไม่ชัดว่า request เข้า leaf หรือ pipeline ใด
7. Pipeline ที่มีหลาย submodels เช่น TableV2 ถูกเรียกว่า leaf ในบางจุด
8. Batch limit ปฏิเสธ request ที่เกินกำหนด แต่ caller บางตัวยังไม่แบ่ง batch

## 4. โครงสร้างเป้าหมาย

การย้ายควรทำทีละส่วน ไม่ควรย้ายทุกไฟล์ใน commit เดียว

```text
models/
|-- services/                         # FastAPI/HTTP transport เท่านั้น
|   |-- gateway/
|   |   `-- main.py
|   |-- leaf/
|   |   |-- layout/main.py
|   |   |-- text_detection/main.py
|   |   |-- text_recognition/main.py
|   |   |-- table_structure/main.py
|   |   `-- siglip/main.py
|   `-- pipeline/
|       |-- document_layout/main.py
|       |-- ocr_custom/main.py
|       |-- ocr_official/main.py
|       |-- table_legacy/main.py
|       |-- table_v2/main.py
|       `-- image_verification/main.py
|
|-- inference/                        # โหลดโมเดลและทำ inference
|   |-- layout_detection.py
|   |-- text_detection.py
|   |-- text_recognition.py
|   |-- table_structure.py
|   |-- table_recognition_v2.py
|   |-- paddle_ocr.py
|   `-- siglip.py
|
|-- pipelines/                        # Workflow และ domain processing
|   |-- ocr/
|   |   |-- orchestrator.py
|   |   |-- polygon_crop.py
|   |   `-- contracts.py
|   |-- layout/
|   |   |-- orchestrator.py
|   |   |-- geometry.py
|   |   `-- contracts.py
|   |-- table/
|   |   |-- orchestrator.py
|   |   |-- grid_analysis.py
|   |   |-- html_parser.py
|   |   |-- cell_assignment.py
|   |   |-- quality.py
|   |   `-- result_assembly.py
|   `-- verification/
|       |-- orchestrator.py
|       `-- scoring.py
|
|-- catalog/                          # Model/version/variant/profile
|   |-- model_variants.json
|   |-- schemas.py
|   `-- resolver.py
|
|-- clients/                          # HTTP client สำหรับ internal services
|   `-- model_service_client.py
|
|-- core/                             # Cross-cutting infrastructure
|   |-- app_factory.py
|   |-- auth.py
|   |-- cache.py
|   |-- config.py
|   |-- contracts.py
|   |-- errors.py
|   |-- image_validation.py
|   |-- limits.py
|   |-- readiness.py
|   |-- request_parsing.py
|   `-- serialization.py
|
|-- requirements/
|-- tests/
|   |-- unit/
|   |-- contract/
|   `-- integration/
|-- deploy/
|-- scripts/
`-- weights/
```

## 5. กฎ Dependency

| Layer | ทำหน้าที่ | Import ได้ | ห้ามทำ |
| --- | --- | --- | --- |
| Gateway | public routing, auth, forwarding | `core`, `clients` | โหลด Paddle, crop ภาพ, resolve path weight |
| HTTP service | parse request, validate input, response envelope | `core`, `pipelines`, `inference` | มี workflow ขนาดใหญ่หรือ hardcode weight path |
| Pipeline | orchestrate หลายขั้นตอน | `clients`, pure processing, contracts | import FastAPI หรือสร้าง HTTP response |
| Inference | initialize/cache/predict โมเดล | `catalog`, adapters, settings | เรียก upstream service หรือ import service layer |
| Catalog | resolve version/variant/path | config schema, filesystem validation | โหลดโมเดลหรือทำ inference |
| Core | utility กลาง | standard library/framework utility | import กลับไปหา service, pipeline หรือ inference |

Dependency ต้องไหลลงทางเดียว:

```text
services -> pipelines/inference -> catalog/core
pipelines -> clients + pure processing -> core
gateway  -> clients -> core
```

## 6. API และ Version Semantics

ใช้ความหมายดังนี้ให้คงที่:

```text
/api/v1       = API contract version
ocr_version   = PP-OCR generation เช่น 5 หรือ 6
profile       = ชุด DET/REC variant ของ pipeline
model         = backward-compatible alias ของ profile/variant
det_model     = override detection variant
rec_model     = override recognition variant
```

ตัวอย่าง TableV2:

```text
POST /api/v1/table-model-results?ocr_version=5&profile=baseline
POST /api/v1/table-model-results?ocr_version=5&profile=thai_ft_v1
POST /api/v1/table-model-results?ocr_version=5&det_model=baseline&rec_model=thai_ft_v1
```

รูปแบบเดิมยังใช้ได้ระหว่าง migration:

```text
POST /api/v1/table-model-results?version=5&model=thai_ft_v1
```

ไม่ใช้ `version=1` แทน Table pipeline generation เพราะชนกับความหมาย OCR generation
และ `/api/v1` ที่มีอยู่แล้ว หากต้องมี pipeline configuration generation จริง ให้เพิ่ม
`pipeline_revision` หรือใช้ชื่อ profile ที่ชัดเจน

## 7. Model Catalog และ Weight Layout

`model_variants.json` ต้องเป็นแหล่งข้อมูลกลางของ:

- model kind
- OCR version
- variant
- official model name
- local model directory
- local weight requirement

โครงสร้าง weight เป้าหมาย:

```text
weights/
|-- detection/
|   |-- det-v5/
|   |   |-- thai_ft_v1/infer/
|   |   `-- thai_ft_v2/infer/
|   `-- det-v6/
|       |-- thai_ft_v1/infer/
|       `-- thai_ft_v2/infer/
|-- recognition/
|   |-- rec-v5/
|   |   |-- thai_ft_v1/infer/
|   |   `-- thai_ft_v2/infer/
|   `-- rec-v6/
|       |-- thai_ft_v1/infer/
|       `-- thai_ft_v2/infer/
|-- layout/
|-- table-wired/
|-- table-wireless/
`-- siglip/
```

กฎของ catalog:

- baseline ใช้ official model cache ได้เมื่อ local directory ว่าง
- non-baseline ต้องมี inference artifacts จริง มิฉะนั้นตอบ
  `MODEL_VARIANT_UNAVAILABLE`
- ห้าม fallback จาก fine-tuned ไป baseline แบบเงียบ
- route และ pipeline ห้าม hardcode path
- การเพิ่ม variant ใหม่ควรแก้ catalog และวาง weight โดยไม่ต้อง duplicate endpoint

## 8. Cache และ GPU Memory

Leaf DET/REC cache ด้วย canonical key:

```text
(version, variant)
```

TableV2 cache ด้วย:

```text
(ocr_version, det_variant, rec_variant)
```

TableV2 หนึ่ง instance มีหลาย submodels จึงไม่ควรตั้ง cache ใหญ่เท่า leaf model ค่าเริ่มต้น:

```env
TABLE_V2_MODEL_CACHE_SIZE=2
```

แนวทางเลือกค่า:

- `1`: ใช้ VRAM ต่ำ แต่สลับ profile แล้วต้องโหลด pipeline ใหม่
- `2`: เหมาะกับ baseline และ active fine-tuned profile หนึ่งชุด
- มากกว่า `2`: ใช้เมื่อวัด VRAM แล้วเท่านั้น

การเอา entry ออกจาก LRU ไม่รับประกันว่า Paddle/CUDA จะคืน VRAM ทันที จึงต้องวัดจาก
process จริง ไม่ควรอาศัยจำนวน cache เพียงอย่างเดียว

## 9. Batch Policy

`MAX_BATCH_IMAGES` เป็น safety limit ของ receiver ไม่ใช่ระบบแบ่ง batch อัตโนมัติ

```text
69 images -> receiver limit 64 -> HTTP 413
```

Caller ที่รู้ลำดับข้อมูลต้องแบ่งเอง:

```text
69 crops -> 64 + 5 -> infer -> concatenate in original order
```

ต้องมี shared batch helper สำหรับ:

- Backend recognition adapter
- Custom OCR polygon crops
- Table/OCR pipeline ที่ส่ง crop batch

Receiver ต้องคง `MAX_BATCH_IMAGES` และ `MAX_BATCH_UPLOAD_MB` ไว้เป็น safety guard

## 10. Refactoring Phases

### Phase 0 — Freeze contracts

- [x] ล็อก inventory และ golden envelope ของทุก public Gateway endpoint ใน `API_V1_CONTRACT.json`
- [x] เพิ่ม golden/contract tests สำหรับ success, error, OpenAPI และ hidden legacy routes
- [ ] รัน `scripts/capture-phase0-baseline.py` บน VPS เพื่อเก็บ response, latency, RAM และ VRAM จริง
- [ ] ตรวจผล legacy endpoint usage จาก log report บน VPS

### Phase 1 — Correctness และ configuration

- [x] แก้ unreachable weight override ใน TableV2
- [x] ให้ TableV2 resolve DET/REC จาก catalog เดียวกับ leaf models
- [x] รองรับ TableV2 `ocr_version/profile` และ alias `version/model`
- [x] รองรับ TableV2 `det_model` และ `rec_model` แยกกัน
- [x] แก้ Gateway default Recognition port เป็น `8004`
- [x] เพิ่ม `REC_SERVICE_URL` ให้ Gateway Compose configuration
- [x] ปรับ catalog/scripts/Compose ให้ใช้ weight layout เดียวกัน
- [x] เพิ่ม batch chunking ใน Custom OCR pipeline โดยรักษาลำดับ polygon
- [x] ส่ง `version/model/det_model/rec_model` ผ่าน full Custom OCR pipeline
- [x] เพิ่ม direct leaf dependencies ใน Gateway readiness

### Phase 2 — แยก Model Loader

- [x] ย้าย Layout loader จาก service ไป `inference/layout_detection.py`
- [x] ย้าย DET/REC loaders จาก `models/` ไป `inference/text_detection.py` และ `inference/text_recognition.py`
- [x] ย้าย Table structure loader ไป `inference/table_structure.py`
- [x] ย้าย SigLIP loader ไป `inference/siglip.py`
- [x] ย้าย PaddleOCR loader ไป `inference/paddle_ocr.py`
- [x] ย้าย TableV2 loader ไป `inference/table_recognition_v2.py`
- [x] ทำ interface ของ inference modules ให้ใช้ `selection_from_settings`, `get_model`, `infer` และ `infer_batch` ตามความเหมาะสม
- [x] เก็บ `models/*.py` เดิมเป็น compatibility wrappers โดยไม่มี loader logic ซ้ำ

รูปแบบเป้าหมาย:

```python
def get_model(selection: ModelSelection) -> object:
    ...


def infer(image_path: str, selection: ModelSelection) -> dict:
    ...
```

### Phase 3 — แยก Pipeline Workflow

- [x] แยก polygon crop ออกจาก OCR orchestration
- [x] แยก legacy response adapter ออกจาก OCR workflow
- [x] ย้าย upstream HTTP calls ไป typed client
- [x] แยก layout geometry ออกจาก remote orchestration
- [x] แยก `table_pipeline.py` ตาม grid, parser, assignment, quality และ assembly
- [x] ทำ pipeline functions ให้รับ dependency ผ่าน argument แทน import global URL

Implementation boundaries and compatibility notes: `pipelines/README.md`.
Validation includes offline workflow, transport, contract and regression tests;
Linux/GPU output and performance comparison remains a deployment gate.

### Phase 4 — แยก Core Infrastructure

- [x] ย้าย `singleflight_lru_cache` จาก `shared/api.py` ไป `core/cache.py`
- [x] แยก request/image parsing ไป `core/request_parsing.py`
- [x] แยก auth และ token validation ไป `core/auth.py`
- [x] แยก request/body/batch limits ไป `core/limits.py`
- [x] แยก readiness helper ไป `core/readiness.py`
- [x] แยก FastAPI factory และ middleware ไป `core/app_factory.py`
- [x] ทำ typed settings แทน `os.getenv()` ที่กระจายอยู่หลายไฟล์

Implementation and settings lifecycle: `core/README.md`.
Validation: 89 tests passed; syntax checked across 53 core/shared/service/inference files.
Compatibility imports are retained in `shared/api.py`. Typed settings cover infrastructure,
service URLs and pipeline runtime options; model-specific configuration stays with its owner.
Intentional fix: clean up already-written images when a later batch image fails validation.
No port, model-cache size, model-selection or inference-pipeline changes in this phase.
Linux/GPU validation remains required before production rollout.

### Phase 5 — Service Topology Cleanup

- [ ] รวม DET v5/v6 เป็น service เดียวถ้า resource test ผ่าน
- [ ] ใช้ `TEXT_DETECTION_URL` เดียวแทน `DET_V5_URL`/`DET_V6_URL`
- [ ] ให้ Gateway ส่ง leaf endpoint ตรงโดยไม่ผ่าน compatibility pipeline
- [ ] เอา recognition-only pass-through ออกจาก OCR Custom หลังหมดผู้ใช้
- [ ] แยก capability discovery/readiness ตาม service ที่เปิดจริง
- [ ] เปลี่ยนชื่อ service ให้สื่อชนิด `leaf` หรือ `pipeline` ชัดเจน

### Phase 6 — Remove deprecated code

- [ ] ลบ generated/runtime files ออกจาก source tree
- [ ] ลบ `models/version_old/` หลังตรวจว่าไม่มี external import
- [ ] ลบ unused local OCR fallback
- [ ] ปิด legacy aliases ด้วย feature flag จริง
- [ ] ลบ pipeline ซ้ำเมื่อมีผลเปรียบเทียบและ migration เสร็จแล้ว

## 11. Requirements Organization

สถานะปัจจุบันจัด version pin ไว้ที่เดียวแล้ว:

```text
requirements/
|-- constraints.txt
|-- base.txt
|-- api.txt
|-- paddle.txt
|-- siglip.txt
|-- demo.txt
|-- dev.txt
`-- all.txt
```

ไฟล์เดิมยังเป็น compatibility wrappers:

```text
requirements.txt
requirements-api.txt
requirements-paddle.txt
requirements-siglip.txt
requirements-dev.txt
```

กฎต่อไป:

- pin version ที่ `requirements/constraints.txt` เท่านั้น
- environment files ระบุเฉพาะ package ที่ต้องใช้
- ห้าม copy version เดียวกันไปหลาย requirements files
- Paddle และ PyTorch ใช้ virtual environment แยกกันบน bare metal
- all-in-one requirements ใช้เฉพาะ Docker image ที่ตั้งใจรวม dependencies
- dependency upgrade ต้องรัน unit tests, import smoke tests และ inference smoke tests

## 12. รายการที่ลบได้และต้อง Deprecate ก่อน

### ลบได้หลังยืนยันว่าไม่ถูก track หรือใช้งาน

```text
**/__pycache__/
*.pyc
.pytest_cache/
.cache/
.run/
.tmp/
logs/*.log
```

ควรอยู่ใน `.gitignore` และไม่ควรถูกส่งขึ้น server เป็น source artifact

### ลบได้หลังตรวจ internal/external references

```text
models/models/version_old/det_old.py
models/models/version_old/rec.py
pipeline/ocr_pipeline.py::predict_custom_ocr
```

Git เป็นที่เก็บประวัติโค้ดเก่า ไม่ควรเก็บไฟล์ `_old` ไว้ใน production source tree

### ต้อง deprecate ก่อนลบ

```text
/predict
/v1/textdetection
/v1/textrecognition
OCR Custom recognition-only pass-through routes
```

ขั้นตอน deprecation:

1. เอาออกจาก discovery/OpenAPI
2. เพิ่ม warning log พร้อม request ID
3. ตรวจ access log ตามช่วงเวลาที่กำหนด
4. แจ้ง client owner
5. ปิดด้วย feature flag
6. ลบใน release ถัดไป

### เลือกจากผลทดสอบก่อนลบ

- `ocr-custom` เทียบกับ `ocr-paddle`
- legacy split table pipeline เทียบกับ TableRecognitionPipelineV2
- DET v5/v6 แยก process เทียบกับ multi-version process เดียว

## 13. Test Strategy

### Unit tests

- model version/variant normalization
- model catalog path resolution
- missing fine-tuned artifacts
- OCR model pair resolution
- polygon ordering และ perspective crop
- table grid/quality/candidate selection
- batch chunking และ result ordering

### Contract tests

- Gateway forwarding fields
- single/batch response envelope
- error code/status/details
- backward-compatible query aliases
- `model_selection` metadata
- legacy route behavior ระหว่าง deprecation

### Integration tests

- Gateway -> REC leaf
- Gateway -> DET leaf
- Gateway -> Custom OCR -> DET/REC
- Gateway -> TableV2 with baseline
- Gateway -> TableV2 with REC-only fine-tune
- batch มากกว่า 64 รายการถูกแบ่งและรวมลำดับถูกต้อง

### GPU smoke tests

- baseline model load
- local fine-tuned model load
- สลับ profile ตาม cache size
- วัด VRAM ก่อน/หลัง eviction
- restart แล้วโหลด model selection ถูกต้อง

## 14. Observability

ทุก inference request ควร log อย่างน้อย:

```text
request_id
service
endpoint
ocr_version
profile
det_variant
rec_variant
det_model_dir หรือ <official-model-cache>
rec_model_dir หรือ <official-model-cache>
image_count
batch_count
duration_ms
```

ห้าม log:

- API token
- รูปภาพ/Base64
- raw request body
- path ภายในใน public error response

## 15. Deployment และ Rollback

แต่ละ phase ต้อง deploy แยกกัน:

1. deploy code ที่รองรับทั้ง old/new config
2. restart service ที่เกี่ยวข้องเท่านั้น
3. ตรวจ readiness และ model selection logs
4. รัน smoke request baseline
5. รัน smoke request fine-tuned
6. ตรวจ RAM/VRAM/latency/error rate
7. ค่อยลบ compatibility code ใน release ภายหลัง

TableV2 และ Gateway บน Linux:

```bash
./scripts/model-stack.sh restart table-v2
./scripts/model-stack.sh restart gateway
./scripts/model-stack.sh readiness table-v2-stack
```

อัปเดต dependencies:

```bash
./scripts/setup-linux.sh paddle
./scripts/setup-linux.sh api
```

Rollback ต้องสามารถทำได้ด้วย source revision เดิมและ catalog เดิม โดยไม่ลบ weight
ระหว่าง deployment

## 16. Definition of Done

งาน refactor แต่ละส่วนถือว่าเสร็จเมื่อ:

- public API เดิมยังทำงานหรือมี deprecation plan ชัดเจน
- unit/contract/integration tests ที่เกี่ยวข้องผ่าน
- ไม่มี model path hardcode ใน route/pipeline
- readiness แสดง dependency ที่ route ใช้งานจริง
- log ระบุ version/variant/local weight ได้
- baseline ไม่ถูกแทนด้วย fine-tune และ fine-tune ไม่ fallback เงียบ
- cache behavior และ VRAM ถูกวัด
- deployment scripts และเอกสารใช้ config เดียวกัน
- ไม่มี generated files หรือ old source files เพิ่มกลับเข้ามา

## 17. ลำดับงานถัดไปที่แนะนำ

งานชุดถัดไปควรทำตามลำดับนี้:

1. เพิ่ม batch chunk helper และใช้ใน Custom OCR
2. ส่ง DET/REC selection ผ่าน full Custom OCR pipeline
3. เพิ่ม TableV2 service tests โดย mock Paddle constructor
4. เพิ่ม Gateway readiness สำหรับ direct REC/DET routes
5. ย้าย model loaders ออกจาก `services/`
6. แยก `shared/api.py`
7. แยก `table_pipeline.py`
8. เริ่ม deprecate redundant routes

อย่าเริ่มจากการ rename/move ทุก directory พร้อมกัน เพราะจะสร้าง diff ใหญ่โดยยังไม่แก้
boundary และ behavior ที่ผิด ควรทำทีละ vertical slice พร้อม tests และ deploy ได้ทุก phase
