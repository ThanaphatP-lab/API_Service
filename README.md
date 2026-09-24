# GPU Model API Platform

ระบบ FastAPI สำหรับ OCR, document layout, table recognition และ SigLIP ที่แยกเป็น 3 ชั้น:

1. **Gateway** เป็น public API URL เดียวที่พอร์ต `8080`
2. **Pipeline services** ต่อผลจากหลายโมเดลและเก็บ business logic
3. **Leaf model services** ทำ inference เพียงโมเดลเดียวและ scale แยกกันได้

แนวทาง response, URI naming และ versioning อ้างอิง [API Design Checklist ของ BorntoDev](https://www.borntodev.com/2026/05/13/api-design-checklist/)

```text
Client → Gateway :8080
            ├─ document-layout pipeline :8010 → layout :8001 + detector :8003
            ├─ custom OCR pipeline :8005   → detector :8003 + recognizer :8004
            ├─ PaddleOCR pipeline :8006    → integrated PaddleOCR
            ├─ table pipeline :8011        → wired :8007 + wireless :8008 + grid analysis + OCR :8005
            └─ verification :8012          → SigLIP :8009
```

## Public API v1

เรียกผ่าน `http://localhost:8080`:

| Method | URI | หน้าที่ | Fields เพิ่มเติม |
| --- | --- | --- | --- |
| GET | `/api/v1/health` | process health | - |
| GET | `/api/v1/readiness` | ตรวจ pipeline และ upstream models | - |
| GET | `/api/v1/services` | รายการ public resources | - |
| POST | `/api/v1/document-layouts` | layout + text ROI | `expand_text_rois`, `auto_roi_mode` |
| POST | `/api/v1/text-detections` | text boxes only (legacy Backend contract) | `version`, `model` |
| POST | `/api/v1/text-detection-batches` | batch text boxes for several ROI crops | `version`, `model` |
| POST | `/api/v1/text-recognitions` | recognition-only for one pre-cropped ROI | `version`, `model` |
| POST | `/api/v1/text-recognition-batches` | recognition-only for pre-cropped ROI batch | `version`, `model` |
| POST | `/api/v1/ocr-results?engine=custom` | OCR แบบต่อ detector/recognizer services | - |
| POST | `/api/v1/ocr-results?engine=paddle` | PaddleOCR integrated pipeline | threshold fields |
| POST | `/api/v1/ocr-result-batches?engine=paddle` | PaddleOCR integrated pipeline สำหรับหลาย ROI | threshold fields |
| POST | `/api/v1/table-results` | wired/wireless selection + OCR | `table_mode`, `include_ocr` |
| POST | `/api/v1/table-model-results` | TableRecognitionPipelineV2 with selectable OCR weights | `ocr_version`/`version`, `profile`/`model` |
| POST | `/api/v1/image-classifications` | raw SigLIP logits สำหรับ Backend-owned scoring | `categories` |
| POST | `/api/v1/image-verifications` | ตรวจ expected category ด้วย SigLIP | `image_category`, `categories` |

Swagger UI: `http://localhost:8080/docs`

Runtime notes:

- Table `auto` uses grid evidence to call wired or wireless first. It calls the alternate model only when the first result is empty or below the original `0.72` quality threshold; the single OCR result is reused.
- `image-verifications` also accepts `image_categories` as a JSON array and evaluates all requested targets from one SigLIP classification. Results are returned in `verifications`.
- `text-detection-batches` and `text-recognition-batches` let Backend batch both OCR stages.
- The application backend uses the split `text-det-v5` + `text-rec-th` path. Detector polygons are perspective-cropped before batched recognition. `ocr-paddle` and `text-det-v6` remain under the `optional-paddle` profile for comparisons.

### Gateway readiness แบบ partial deployment

Gateway ตรวจเฉพาะ pipeline และ direct leaf ที่เปิดใช้งานจริงได้ด้วย environment variables:

```text
GATEWAY_ENABLED_PIPELINES=ocr-custom,image-verification,text-det-v5,text-recognition,siglip
GATEWAY_REQUIRED_PIPELINES=ocr-custom,text-det-v5,text-recognition
GATEWAY_READINESS_TIMEOUT_SECONDS=10
```

ชื่อ pipeline ที่รองรับคือ `layout`, `ocr-custom`, `ocr-paddle`, `table`,
`table-model` และ `image-verification`; direct leaf คือ `text-det-v5`,
`text-det-v6`, `text-recognition` และ `siglip` หากไม่กำหนด
`GATEWAY_ENABLED_PIPELINES` จะตรวจทั้งหมด

- ทุก pipeline พร้อม: HTTP `200`, status `ready`
- มีบาง optional pipeline ไม่พร้อม: HTTP `200`, status `degraded`
- required pipeline ไม่พร้อม หรือไม่มี pipeline ใดพร้อมเลย: HTTP `503`

ตัวอย่างสำหรับ Server ที่เปิดเฉพาะ Custom OCR:

```cmd
set GATEWAY_ENABLED_PIPELINES=ocr-custom,text-det-v5,text-recognition
set GATEWAY_REQUIRED_PIPELINES=ocr-custom,text-det-v5,text-recognition
scripts\run-service.cmd gateway
```

ผล readiness จะแสดงสถานะและเวลาตอบสนองแยกแต่ละ pipeline เพื่อให้ debug ได้โดยไม่ทำให้ระบบทั้งหมด offline เพราะ optional service เพียงตัวเดียว

Custom OCR รองรับการเลือก DET/REC variant ตลอดทั้ง pipeline และแบ่ง polygon
crops ก่อนส่งเข้า Recognition leaf โดยอัตโนมัติ:

```text
POST /api/v1/ocr-results?engine=custom&version=6&model=thai_ft_v1
POST /api/v1/ocr-results?engine=custom&version=6&det_model=baseline&rec_model=thai_ft_v1
```

`OCR_RECOGNITION_BATCH_SIZE` มีค่าเริ่มต้น `64` และต้องไม่มากกว่า
`MAX_BATCH_IMAGES` ของ Recognition service ลำดับผลลัพธ์หลังรวม batch จะเหมือนลำดับ polygon เดิม

## Input ที่รองรับ

ทุก image endpoint รองรับทั้ง multipart file และ JSON Base64/Data URL

Multipart:

```cmd
curl -X POST "http://localhost:8080/api/v1/ocr-results?engine=custom" -F "image=@sample.jpg"
```

JSON Base64:

```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAA..."
}
```

ตัวอย่าง Document Layout:

```cmd
curl -X POST http://localhost:8080/api/v1/document-layouts ^
  -F "image=@sample.jpg" ^
  -F "expand_text_rois=true" ^
  -F "auto_roi_mode=text-line"
```

`auto_roi_mode` รองรับ `text-line`, `layout`, `hybrid`

ตัวอย่าง PaddleOCR parameters:

```cmd
curl -X POST "http://localhost:8080/api/v1/ocr-results?engine=paddle" ^
  -F "image=@sample.jpg" ^
  -F "text_det_unclip_ratio=2" ^
  -F "text_det_thresh=0.25" ^
  -F "text_det_box_thresh=0.6"
```

ตัวอย่าง Table pipeline:

```cmd
curl -X POST http://localhost:8080/api/v1/table-results ^
  -F "image=@table.jpg" ^
  -F "table_mode=auto" ^
  -F "include_ocr=true"
```

`table_mode` รองรับ `auto`, `wired`, `wireless`

ตัวอย่าง SigLIP verification:

```cmd
curl -X POST http://localhost:8080/api/v1/image-verifications ^
  -F "image=@qr.png" ^
  -F "image_category=qr_code"
```

ส่ง categories เองได้ทั้ง JSON body หรือ JSON string ใน multipart:

```json
{
  "image": "data:image/png;base64,...",
  "image_category": "qr_code",
  "categories": [
    {
      "value": "qr_code",
      "label": "QR Code",
      "prompt": "This is a photo of a QR code.",
      "match_threshold": 0.55,
      "margin_threshold": 0.05,
      "enabled": true
    }
  ]
}
```

## Success response

ทุก endpoint ใช้โครงสร้างเดียวกัน:

```json
{
  "data": {
    "predictions": []
  },
  "meta": {
    "request_id": "req_5e5e40b2...",
    "api_version": "v1",
    "service": "text-detection-model",
    "model": "PP-OCRv6_medium_det",
    "duration_ms": 121.4
  }
}
```

ส่ง `X-Request-ID` เองได้ หากไม่ส่งระบบจะสร้าง `req_<uuid>` และคืนใน body/header

## Error response ที่ใช้ debug

```json
{
  "error": {
    "code": "INVALID_IMAGE_BASE64",
    "message": "The image field is not valid Base64.",
    "details": [
      {
        "field": "image",
        "issue": "invalid_base64"
      }
    ],
    "request_id": "req_5e5e40b2...",
    "docs": "http://localhost:8080/docs"
  }
}
```

หลักการ debug:

- Client ใช้ `error.code` ทำ logic ไม่ parse จาก message
- `details` ระบุ field, issue, received value หรือ upstream ที่มีปัญหา
- `request_id` เชื่อม response กับ server log และถูกส่งต่อไปทุก upstream
- `UPSTREAM_MODEL_ERROR` มี upstream status/code/message/request ID
- Production ไม่คืน stack trace; server log เก็บ traceback พร้อม request ID
- ตั้ง `DEBUG_ERRORS=true` ได้เฉพาะ development เพื่อเพิ่ม exception type/reason

Status ที่ใช้บ่อย:

| Status | ตัวอย่าง code |
| ---: | --- |
| 400 | `INVALID_JSON`, `BAD_REQUEST` |
| 413 | `PAYLOAD_TOO_LARGE` |
| 415 | `UNSUPPORTED_MEDIA_TYPE` |
| 422 | `VALIDATION_ERROR`, `INVALID_IMAGE`, `IMAGE_REQUIRED` |
| 429 | `RATE_LIMIT_EXCEEDED` |
| 502 | `UPSTREAM_MODEL_ERROR`, `UPSTREAM_CONTRACT_MISMATCH` |
| 503 | `MODEL_NOT_READY`, `GPU_UNAVAILABLE`, `UPSTREAM_UNAVAILABLE` |
| 504 | `UPSTREAM_TIMEOUT` |

## Rate limit

ค่าเริ่มต้นคือ 120 requests ต่อ 60 วินาทีต่อ client IP:

```env
RATE_LIMIT_REQUESTS=120
RATE_LIMIT_WINDOW_SECONDS=60
MAX_UPLOAD_MB=20
CORS_ORIGINS=http://localhost:3000,https://your-frontend.example.com
```

ตั้ง `RATE_LIMIT_REQUESTS=0` เพื่อปิด limiter ใน development

Response headers:

```text
X-RateLimit-Limit
X-RateLimit-Remaining
X-RateLimit-Reset
Retry-After          # เมื่อเป็น HTTP 429
```

Limiter ในตัว service เป็น in-memory ต่อ process สำหรับป้องกันพื้นฐาน หาก production มีหลาย replica ต้องวาง distributed rate limit ที่ Gateway/Ingress เช่น Redis, NGINX, Kong หรือ API Gateway ของ cloud เพื่อให้ quota รวมทุก replica ถูกต้อง

ตั้ง `TRUST_PROXY_HEADERS=true` เมื่อ API อยู่หลัง trusted reverse proxy เท่านั้น เพื่อใช้ IP จาก `X-Forwarded-For`

## Authentication และ production security

ระบบแยก secret เป็น 2 ขอบเขต ห้ามใช้ค่าเดียวกัน:

```text
Backend -- MODEL_GATEWAY_API_KEY --> Gateway
Gateway/Pipeline -- INTERNAL_API_TOKEN --> Pipeline/Leaf Model
```

ส่ง token ผ่าน header เท่านั้น ไม่ใส่ใน URL หรือ query string:

```http
Authorization: Bearer <MODEL_GATEWAY_API_KEY>
```

รองรับ `X-API-Key` เพื่อ compatibility แต่แนะนำ Bearer header สำหรับ Backend
client ตัวหลัก ใน `APP_ENV=production` token ต้องยาวอย่างน้อย 32 ตัวอักษร,
ห้ามเป็น placeholder และ application จะไม่ start หากขาด token

สร้าง secret คนละค่า:

```cmd
python -c "import secrets; print(secrets.token_urlsafe(48))"
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

เริ่มจากไฟล์ `.env.production.example` และเก็บค่าจริงใน secret manager ของ
server ไม่ commit `.env` ลง Git Backend ต้องตั้งค่าคู่กันดังนี้:

```env
MODEL_GATEWAY_URL=http://127.0.0.1:8080
MODEL_GATEWAY_API_KEY=<same value used by Gateway>
MODEL_GATEWAY_AUTH_REQUIRED=true
```

การ rotate แบบไม่ตัดการเชื่อมต่อทำเป็น 3 ระยะ: เพิ่ม secret ใหม่ไว้ในตัวแปร
`*_PREVIOUS` ของผู้รับก่อน, สลับตัวแปรหลักของทั้งผู้ส่งและผู้รับให้เป็นค่าใหม่
พร้อมเก็บค่าเก่าไว้ใน `*_PREVIOUS`, แล้วลบ `_PREVIOUS` หลังทุก instance ใช้
ค่าใหม่ครบเพื่อ revoke secret เก่า

`GET /health` เปิดไว้สำหรับ liveness โดยไม่ต้องใช้ token ส่วน readiness,
discovery และ inference ต้องยืนยัน token เมื่อกำหนด secret แล้ว Production จะ
ปิด `/docs`, `/redoc`, `/openapi.json` และ legacy `/predict` โดยอัตโนมัติ

Resource controls:

```env
MAX_REQUEST_MB=28
MAX_UPLOAD_MB=20
MAX_RATE_LIMIT_KEYS=10000
MAX_IMAGE_PIXELS=40000000
MAX_IMAGE_DIMENSION=12000
MAX_BATCH_IMAGES=64
MAX_BATCH_UPLOAD_MB=64
MAX_CONCURRENT_REQUESTS=1
INFERENCE_QUEUE_TIMEOUT_SECONDS=5
```

Application limiter เป็น defense in depth; public edge ควรมี body/rate/
connection limits อีกชั้นหนึ่ง

## HTTPS ด้วย Nginx

ใช้ Nginx ทำ TLS termination ได้ โดยให้ public traffic เข้า `443` แล้ว proxy
ไปยัง Gateway `127.0.0.1:8080` เท่านั้น ตัวอย่างพร้อม TLS protocols, HSTS,
security headers, upload limit และ edge rate limit อยู่ที่:

```text
deploy/nginx/nginx.conf
deploy/nginx/README.md
```

เมื่อ Nginx อยู่เครื่องเดียวกันให้ตั้ง:

```env
GATEWAY_HOST=127.0.0.1
INTERNAL_SERVICE_HOST=127.0.0.1
GATEWAY_BIND_ADDRESS=127.0.0.1
DEMO_BIND_ADDRESS=127.0.0.1
TRUST_PROXY_HEADERS=true
```

URL ภายนอกจึงเป็น `https://api.example.com/api/v1/...` และ firewall ต้องไม่
เปิดพอร์ต `8001-8012` หรือ `8080` สู่ Internet

## Ports

| Port | Service | Type |
| ---: | --- | --- |
| 8080 | Public API gateway | Gateway |
| 8010 | document-layout | Pipeline |
| 8005 | custom OCR | Pipeline |
| 8006 | PaddleOCR | Pipeline/GPU |
| 8011 | table recognition | Pipeline |
| 8012 | image verification | Pipeline |
| 8013 | TableRecognitionPipelineV2 | Pipeline/GPU |
| 8001 | PP-DocLayoutV3 | Leaf/GPU |
| 8002 | PP-OCRv5 server det | Leaf/GPU |
| 8003 | PP-OCRv6 medium det | Leaf/GPU |
| 8004 | Thai recognition | Leaf/GPU |
| 8007 | SLANeXt wired | Leaf/GPU |
| 8008 | SLANeXt wireless | Leaf/GPU |
| 8009 | SigLIP | Leaf/GPU |
| 8501 | Streamlit demo | Demo |

ตารางนี้เป็น runtime port map สำหรับ debug/non-Docker เท่านั้น ใน production
พอร์ต Pipeline/Leaf bind ที่ loopback และ Docker Compose ไม่ publish พอร์ตเหล่านี้
ออก host; Gateway และ Demo publish เฉพาะ loopback โดยค่าเริ่มต้น หากต้องเปิดให้
network ภายในเข้าถึงโดยตรงจึงค่อยกำหนด bind address เอง

Leaf model endpoints:

```text
POST :8001/api/v1/layout-predictions
POST :8002/api/v1/text-detections
POST :8003/api/v1/text-detections
POST :8004/api/v1/text-recognitions
POST :8004/api/v1/text-recognition-batches
POST :8007/api/v1/table-structures
POST :8008/api/v1/table-structures
POST :8009/api/v1/image-classifications
```

`/predict` และ `/health` แบบเก่ายังเก็บเป็น hidden compatibility aliases ใน leaf services ระหว่างย้ายระบบ แต่โค้ดใหม่ควรเรียก `/api/v1/...`; production ปิด `/predict` เว้นแต่ตั้ง `LEGACY_ENDPOINTS_ENABLED=true`

## รันด้วย Docker

ต้องมี NVIDIA driver, Docker Desktop/Engine และ NVIDIA Container Toolkit หรือ WSL2 GPU support

```cmd
cd models
docker compose up --build
```

### สร้างชุด v2 ใหม่โดยไม่แตะ container/image ชุดเก่า

คัดลอก `.env.v2.example` เป็น `.env.v2` แล้วใช้ Compose project name ใหม่:

```cmd
copy .env.v2.example .env.v2
docker compose -p api-service-v2 --env-file .env.v2 up -d --build
```

ชุดใหม่ใช้ image `api-service-model-v2:latest`, network/project `api-service-v2` และพอร์ต `18xxx` จึงรันคู่กับชุดเก่าได้:

```text
Gateway v2: http://localhost:18080/docs
Demo v2:    http://localhost:18501
```

ตรวจเฉพาะชุดใหม่:

```cmd
docker compose -p api-service-v2 --env-file .env.v2 ps
docker compose -p api-service-v2 --env-file .env.v2 logs -f gateway
```

เปิด:

```text
Development Swagger: http://localhost:8080/docs
Demo:           http://localhost:8501
```

ดูสถานะ:

```cmd
docker compose ps
docker compose logs -f gateway
docker compose logs -f text-det-v6 text-rec-th ocr-custom
```

สร้างใหม่หลังแก้ shared code:

```cmd
docker compose build model-image
docker compose up -d --force-recreate
```

## รันบน Windows CMD โดยไม่ใช้ Docker

เตรียม environment:

```cmd
cd models
scripts\setup-local.cmd
```

หรือเลือกติดตั้งเฉพาะกลุ่มเพื่อลดเวลาและพื้นที่:

```cmd
scripts\setup-local.cmd paddle
scripts\setup-local.cmd siglip
scripts\setup-local.cmd api
```

คำสั่งนี้สร้าง environment แยกเพื่อป้องกัน cuDNN DLL conflict บน Windows:

```text
.venv-paddle312  = PaddleOCR/PP-DocLayout/SLANeXt
.venv-siglip312  = PyTorch/Transformers/SigLIP
.venv-api312     = Gateway/Pipelines/Streamlit
```

เปิด CMD แยกหน้าต่างแล้วรัน leaf services ก่อน:

```cmd
scripts\run-service.cmd layout
scripts\run-service.cmd det-v6
scripts\run-service.cmd rec-th
scripts\run-service.cmd table-wired
scripts\run-service.cmd table-wireless
scripts\run-service.cmd siglip
```

จากนั้นเปิด pipeline services:

```cmd
scripts\run-service.cmd ocr-custom
scripts\run-service.cmd ocr-paddle
scripts\run-service.cmd layout-pipeline
scripts\run-service.cmd table-pipeline
scripts\run-service.cmd image-verification
```

สุดท้ายเปิด Gateway และ Demo:

```cmd
scripts\run-service.cmd gateway
scripts\run-demo.cmd
```

## Local weights

```text
weights/
  layout/
  detection/
    det-v5/
      thai_ft_v1/infer/
      thai_ft_v2/infer/
    det-v6/
      thai_ft_v1/infer/
      thai_ft_v2/infer/
  recognition/
    rec-v5/
      thai_ft_v1/infer/
      thai_ft_v2/infer/
    rec-v6/
      thai_ft_v1/infer/
      thai_ft_v2/infer/
  table-wired/
  table-wireless/
  siglip/
```

### DET/REC model variants

Both text detection and text recognition keep their existing services and ports.
The optional query parameter `model` defaults to `baseline`; `version` accepts
`5`, `v5`, `6`, or `v6`.

```text
POST /api/v1/text-detections?version=5
POST /api/v1/text-detections?version=6&model=thai_ft_v1
POST /api/v1/text-recognitions?version=5&model=baseline
POST /api/v1/text-recognitions?version=6&model=thai_ft_v2
```

Variant-to-weight mappings live in `model_variants.json`. The example mapping is:

```text
thai_ft_v1 -> thai_ft_v1/infer
thai_ft_v2 -> thai_ft_v2/infer
```

Copy the complete Paddle inference export into the selected `infer` directory.
An empty directory (or one containing only `.gitkeep`) returns HTTP 422
`MODEL_VARIANT_UNAVAILABLE`; it never silently falls back to baseline. Baseline
continues to resolve the official model by name when its local weight directory
is empty. Loaded model instances are cached by `(version, model)` and reused.

### TableRecognitionPipelineV2 OCR variants

TableV2 resolves its internal text detector and recognizer from the same
`model_variants.json` catalog as the standalone leaf services. The clearer
parameter names are `ocr_version` and `profile`; `version` and `model` remain
supported as backward-compatible aliases.

```text
POST /api/v1/table-model-results?ocr_version=5&profile=baseline
POST /api/v1/table-model-results?ocr_version=5&profile=thai_ft_v1
POST /api/v1/table-model-results?version=6&model=thai_ft_v2
POST /api/v1/table-model-results?ocr_version=5&det_model=baseline&rec_model=thai_ft_v1
```

For a non-baseline profile, both the configured detection and recognition
weight directories must contain exported inference artifacts. TableV2 loads
those files inside its own process; it does not call the DET/REC leaf services.
Use `det_model` and `rec_model` when only one component should change; `profile`
or `model` remains a shorthand that selects the same variant for both.
`TABLE_V2_MODEL_CACHE_SIZE` defaults to `2` to avoid retaining too many complete
pipeline instances in GPU memory.

## Inference layer

Model constructors และ inference logic อยู่ใน `inference/` ส่วน
`services/*/main.py` รับผิดชอบเฉพาะ HTTP parsing, validation และ response
envelope โมดูล inference ใช้ interface หลักร่วมกัน:

```python
selection = selection_from_settings(...)
runtime = get_model(selection)
payload = infer(image_path, selection)
```

DET, REC และ TableV2 files เดิมใต้ `models/` เป็น compatibility wrappers
ชั่วคราว โค้ดใหม่ต้อง import จาก `inference.*` รายละเอียด boundary อยู่ที่
`inference/README.md`

## Dependency groups

Pinned versions live only in `requirements/constraints.txt`. Installable groups
live under `requirements/` (`api.txt`, `paddle.txt`, `siglip.txt`, `demo.txt`,
and `dev.txt`). The root `requirements-*.txt` files are compatibility wrappers
for existing setup scripts.

## Linux server without Docker

For a bare-metal Linux server, use the process management scripts and deployment
guide in `deploy/linux/README.md`:

```bash
chmod +x scripts/setup-linux.sh scripts/model-stack.sh
./scripts/setup-linux.sh all
cp .env.linux.example .env.runtime
./scripts/model-stack.sh check
./scripts/model-stack.sh start core-stack
./scripts/model-stack.sh start ocr-custom-stack
./scripts/model-stack.sh status all
./scripts/model-stack.sh readiness ocr-custom-stack
./scripts/model-stack.sh stop all
```

The scripts bind services to loopback, preserve processes after SSH logout,
store one PID per service under `.run/`, and write logs under `logs/`. Use
systemd/Supervisor for automatic start after reboot and crash recovery.

ถ้า directory ว่าง ระบบจะ resolve/download official model ด้วยชื่อโมเดล หากมีไฟล์ local export ระบบจะใช้ directory ที่กำหนดผ่าน environment

## Tests

Phase 4 separates API infrastructure into `core/`, with typed infrastructure
settings and compatibility imports in `shared/api.py`.
See [core boundaries and settings lifecycle](core/README.md).

Phase 3 separates service routes, injectable upstream clients, pipeline workflows
and geometry/parsing/scoring. See [pipeline boundaries](pipelines/README.md).
The old `pipeline/` imports remain compatibility wrappers; API routes and model
selection behavior are unchanged by this phase.

```cmd
.venv-api312\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv-api312\Scripts\python.exe -m pytest -q tests
python -m compileall -q shared models pipeline services demo
docker compose config --quiet
```

Tests ครอบคลุม success/error contract, Base64 input, request ID, rate limit และ image-verification threshold/margin

ก่อนเริ่ม refactor phase ถัดไป ใช้ `API_V1_CONTRACT.json` เป็น public route contract
และทำตาม `PHASE0_BASELINE.md` เพื่อเก็บ latency, RAM/VRAM และ legacy-route usage
จาก Linux/GPU runtime จริง

## การเพิ่ม API v2 หรือเปลี่ยนโมเดล

- ห้ามเปลี่ยน schema ของ `/api/v1` แบบ breaking change
- เพิ่ม router/service ใหม่ใต้ `/api/v2`
- ให้ Gateway route v1/v2 ไป pipeline คนละ deployment ได้
- Leaf service เลือก model/weights ผ่าน `*_MODEL_NAME`, `*_MODEL_DIR`, `MODEL_VERSION`
- Pipeline อ้างอิง leaf ด้วย URL จึงสลับ v1/v2 หรือ scale replica โดยไม่แก้ client
- Production ควรใส่ model version หรือ weights checksum ลง `MODEL_VERSION`
