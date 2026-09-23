# Model API security and deployment boundary

เอกสารนี้สรุป security boundary ของ model platform ตาม OWASP API Security
Top 10 (2023) และ OWASP REST Security Cheat Sheet

แหล่งอ้างอิง:

- https://owasp.org/API-Security/editions/2023/en/0x11-t10/
- https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html

## Trust boundary

```text
Browser -- user access token --> Backend
Backend -- MODEL_GATEWAY_API_KEY --> Model Gateway
Gateway/Pipeline -- INTERNAL_API_TOKEN --> Pipeline/Leaf models
Internet -- HTTPS 443 --> Nginx -- HTTP loopback --> Gateway 127.0.0.1:8080
```

Model Gateway ไม่ได้แทน user authentication หรือ user authorization ของ Backend
เพราะ service token ระบุได้เพียงว่า caller คือ Backend ที่ไว้ใจ ไม่ได้ระบุว่า user
คนใดมีสิทธิ์ใช้ object ใด Backend จึงต้องตรวจ user token, role และ object ownership
ก่อนเรียก Gateway ทุกครั้ง

## Controls implemented

- Protected endpoints รับ `Authorization: Bearer` และรองรับ `X-API-Key` สำหรับ
  compatibility โดยไม่รับ secret ผ่าน URL/query string
- ใช้ secret คนละค่าระหว่าง public Gateway และ internal services, เปรียบเทียบแบบ
  constant-time และ fail startup เมื่อ production ไม่มี secret ที่ปลอดภัย
- `/health` เปิดสำหรับ liveness; readiness, discovery และ inference ต้องมี token
- รองรับ secondary secret ชั่วคราวผ่านตัวแปร `*_PREVIOUS` สำหรับ rotation/revocation
- จำกัด request/image/batch bytes, image dimensions/pixels, request rate,
  concurrent inference และ queue wait
- จำกัดจำนวน key ใน in-process limiter เพื่อไม่ให้ตาราง rate-limit โตไม่สิ้นสุด
- ตรวจ Content-Type, decode รูปจริง และ validate/range-check endpoint fields
- ใช้ response contract ที่มี code, message, details และ request_id; public Gateway
  ซ่อน internal URL/path/exception แต่บันทึกรายละเอียดไว้ใน server log
- ปิด docs/OpenAPI และ legacy `/predict` ใน production; endpoint ใหม่อยู่ใต้ `/api/v1`
- CORS ปิดโดย default และ Docker/non-Docker internal services bind เฉพาะ private
  network หรือ loopback
- Nginx บังคับ HTTP ไป HTTPS, TLS 1.2/1.3, body/rate/connection limits,
  canonical host, security headers และ JSON errors สำหรับ edge 413/429

## Production requirements

1. ตั้ง `APP_ENV=production` และสร้าง secret สุ่มสองค่าที่ต่างกันอย่างน้อย 32 ตัวอักษร
2. เก็บ secret ใน secret manager หรือ environment ของ process manager ห้าม commit
   `.env` จริง และห้ามเขียน token ลง access/application logs
3. ให้ Backend เรียก Gateway ผ่าน private/loopback URL และให้ public traffic ผ่าน
   Nginx/ingress ที่ HTTPS เท่านั้น
4. ปิด firewall ของพอร์ต Leaf/Pipeline และพอร์ต Gateway 8080 จาก Internet
5. เปิด `TRUST_PROXY_HEADERS=true` เฉพาะเมื่อ client ข้าม trusted Nginx ไม่ได้
6. จัดการ certificate issuance/renewal และทดสอบ Nginx config ก่อน reload
7. ส่ง `request_id` ไปตลอด chain และทำ alert จาก 401, 429, 5xx และ readiness failure
8. หากรันหลาย process/server ให้ใช้ edge/API gateway หรือ distributed limiter
   เพราะ application limiter เป็น per-process defense in depth

## Known blocker outside the model folder

ณ เวลาที่ตรวจ `project_backend/app/routes.py` ยังไม่ได้ผูก auth dependency กับหลาย
route และ `project_backend/app/auth_service.py::current_user` มีพฤติกรรม fail-open
เป็น mock user เมื่อไม่มี/ผิด token ดังนั้นห้ามถือว่า public Backend ปลอดภัยสำหรับ
production จนกว่าจะกำหนด authorization policy ของแต่ละ route แล้วเปลี่ยนเป็น
fail-closed การแก้จุดนี้มีผลต่อสิทธิ์ user/role และ object ownership ของทั้งผลิตภัณฑ์
จึงต้องทำเป็นงาน Backend security แยกจาก service-to-service auth ใน model platform
