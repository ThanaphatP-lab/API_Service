# Phase 0 Contract and Runtime Baseline

เอกสารนี้ใช้ล็อกพฤติกรรมก่อน refactor โดยแยกข้อมูลออกเป็นสองส่วน:

1. Source contract ที่ตรวจได้ใน CI โดยไม่ต้องโหลด Paddle/GPU
2. Runtime baseline ที่ต้องเก็บจาก Linux server ขณะเปิด service จริง

## Source contract

ไฟล์ `API_V1_CONTRACT.json` เป็น inventory ของ public Gateway API v1 และกำหนด
field ขั้นต่ำของ success/error envelope ส่วน hidden legacy routes ถูกบันทึกแยกเพื่อ
ป้องกันไม่ให้กลับเข้า OpenAPI โดยไม่ตั้งใจ

Contract tests อยู่ใน:

```text
tests/test_public_api_contract.py
tests/test_gateway_readiness.py
tests/test_ocr_pipeline.py
```

เมื่อเพิ่ม ลบ หรือเปลี่ยน public route ต้องแก้ contract file และ test พร้อมเหตุผล
ห้าม update golden contract เพียงเพื่อทำให้ test ผ่านโดยไม่ตรวจ backward compatibility

## เก็บ runtime baseline บน VPS

รันหลังจากเปิด stack และ readiness ของ service ที่ต้องการวัดพร้อมแล้ว:

```bash
cd /workspace/API_service/models
source .venv-api/bin/activate
python scripts/capture-phase0-baseline.py \
  --gateway http://127.0.0.1:8080 \
  --image /absolute/path/to/sample.png \
  --logs logs
```

หากเป็น partial deployment และตั้งใจให้ optional endpoints ตอบ error ให้เพิ่ม
`--allow-errors` เพื่อเก็บผลทั้งหมดโดยไม่ให้ process จบด้วย exit code 1:

```bash
python scripts/capture-phase0-baseline.py \
  --image /absolute/path/to/sample.png \
  --allow-errors
```

Token อ่านจาก `MODEL_GATEWAY_API_KEY` โดยอัตโนมัติ หรือส่งผ่าน `--token` ได้
แต่แนะนำให้ใช้ environment variable เพื่อไม่ให้ secret ติด shell history

ผลลัพธ์เริ่มต้นอยู่ที่ `baselines/phase0-<UTC timestamp>.json` และประกอบด้วย:

- status, response body, request ID และ end-to-end latency ของทุก public path
- RSS ของ process จาก `ps`
- GPU memory ต่อ process จาก `nvidia-smi`
- จำนวนการเรียก `/predict`, `/v1/textdetection`, `/v1/textrecognition` และ `/health`
  ที่พบใน `logs/*.log`

ไฟล์ baseline อาจมีข้อมูลผล inference ของเอกสารตัวอย่าง จึงต้องตรวจข้อมูลก่อน commit
และไม่ควรใช้ภาพ production ที่มีข้อมูลส่วนบุคคล

## เกณฑ์ก่อนจบ Phase 0

- Contract tests ผ่านใน API virtual environment
- เก็บ baseline จาก sample เดิมอย่างน้อย 3 รอบหลัง warm-up
- บันทึก GPU รุ่น, driver, Paddle version และจำนวน service ที่เปิดพร้อมผล baseline
- ตรวจ legacy access log ในช่วงเวลาที่เป็นตัวแทน traffic จริง
- ระบุ legacy route ที่ยังมี client ใช้งานก่อนเริ่ม deprecation
