# Inference layer

โมดูลในโฟลเดอร์นี้รับผิดชอบเฉพาะการเลือก configuration, โหลด/cache โมเดล,
เรียก inference และแปลงผลลัพธ์เป็น leaf contract เท่านั้น ห้าม import FastAPI,
สร้าง HTTP response หรือเรียก upstream service

Interface หลักของแต่ละโมดูลมีรูปแบบเดียวกัน:

```python
selection = selection_from_settings(...)
runtime = get_model(selection)
payload = infer(image_path, selection)
```

โมดูลที่รองรับหลายภาพเพิ่ม `infer_batch(image_paths, selection)` โดยยังคง
`get_model()` เป็นจุดเดียวที่สร้างและ cache model runtime

| Module | Runtime |
| --- | --- |
| `layout_detection.py` | `LayoutDetection` |
| `text_detection.py` | `TextDetection` |
| `text_recognition.py` | `TextRecognition` |
| `table_structure.py` | `TableStructureRecognition` |
| `paddle_ocr.py` | integrated `PaddleOCR` |
| `table_recognition_v2.py` | `TableRecognitionPipelineV2` |
| `siglip.py` | SigLIP processor/model runtime |

Old models/ wrappers were removed in Phase 6. Import the owning inference module directly.
