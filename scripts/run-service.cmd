@echo off
setlocal EnableExtensions

rem Usage: scripts\run-service.cmd SERVICE
rem Services: layout detection rec-th ocr-custom ocr-paddle table-wired table-wireless table-v2 siglip layout-pipeline table-pipeline image-verification gateway

set "SERVICE=%~1"
set "ROOT=%~dp0.."
cd /d "%ROOT%"

if "%SERVICE%"=="" goto :usage

set "PYTHON=.venv-api312\Scripts\python.exe"
if /I "%SERVICE%"=="layout" set "PYTHON=.venv-paddle312\Scripts\python.exe"
if /I "%SERVICE%"=="detection" set "PYTHON=.venv-paddle312\Scripts\python.exe"
if /I "%SERVICE%"=="rec-th" set "PYTHON=.venv-paddle312\Scripts\python.exe"
if /I "%SERVICE%"=="ocr-paddle" set "PYTHON=.venv-paddle312\Scripts\python.exe"
if /I "%SERVICE%"=="table-wired" set "PYTHON=.venv-paddle312\Scripts\python.exe"
if /I "%SERVICE%"=="table-wireless" set "PYTHON=.venv-paddle312\Scripts\python.exe"
if /I "%SERVICE%"=="table-v2" set "PYTHON=.venv-paddle312\Scripts\python.exe"
if /I "%SERVICE%"=="siglip" set "PYTHON=.venv-siglip312\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo [ERROR] Required virtual environment not found: %PYTHON%
  echo Run: scripts\setup-local.cmd
  exit /b 1
)

if "%MODEL_DEVICE%"=="" set "MODEL_DEVICE=gpu:0"
if "%PADDLE_PDX_MODEL_SOURCE%"=="" set "PADDLE_PDX_MODEL_SOURCE=BOS"
if "%RATE_LIMIT_REQUESTS%"=="" set "RATE_LIMIT_REQUESTS=120"
if "%RATE_LIMIT_WINDOW_SECONDS%"=="" set "RATE_LIMIT_WINDOW_SECONDS=60"
if "%MAX_REQUEST_MB%"=="" set "MAX_REQUEST_MB=28"
if "%MAX_UPLOAD_MB%"=="" set "MAX_UPLOAD_MB=20"
if "%MAX_CONCURRENT_REQUESTS%"=="" set "MAX_CONCURRENT_REQUESTS=1"
if /I "%SERVICE%"=="gateway" if "%GATEWAY_MAX_CONCURRENT_REQUESTS%"=="" set "MAX_CONCURRENT_REQUESTS=4"
if /I "%SERVICE%"=="gateway" if not "%GATEWAY_MAX_CONCURRENT_REQUESTS%"=="" set "MAX_CONCURRENT_REQUESTS=%GATEWAY_MAX_CONCURRENT_REQUESTS%"
if "%INTERNAL_SERVICE_HOST%"=="" set "INTERNAL_SERVICE_HOST=127.0.0.1"
if "%GATEWAY_HOST%"=="" set "GATEWAY_HOST=127.0.0.1"
if "%LAYOUT_PORT%"=="" set "LAYOUT_PORT=8001"
if "%DETECTION_PORT%"=="" set "DETECTION_PORT=8002"
if "%TEXT_DETECTION_URL%"=="" set "TEXT_DETECTION_URL=http://127.0.0.1:%DETECTION_PORT%"
set "DET_SERVICE_URL=%TEXT_DETECTION_URL%"
if "%REC_TH_PORT%"=="" set "REC_TH_PORT=8004"
if "%OCR_CUSTOM_PORT%"=="" set "OCR_CUSTOM_PORT=8005"
if "%OCR_PADDLE_PORT%"=="" set "OCR_PADDLE_PORT=8006"
if "%TABLE_WIRED_PORT%"=="" set "TABLE_WIRED_PORT=8007"
if "%TABLE_WIRELESS_PORT%"=="" set "TABLE_WIRELESS_PORT=8008"
if "%SIGLIP_PORT%"=="" set "SIGLIP_PORT=8009"
if "%LAYOUT_PIPELINE_PORT%"=="" set "LAYOUT_PIPELINE_PORT=8010"
if "%TABLE_PIPELINE_PORT%"=="" set "TABLE_PIPELINE_PORT=8011"
if "%IMAGE_VERIFICATION_PORT%"=="" set "IMAGE_VERIFICATION_PORT=8012"
if "%TABLE_V2_PORT%"=="" set "TABLE_V2_PORT=8013"
if "%GATEWAY_PORT%"=="" set "GATEWAY_PORT=8080"

if /I "%SERVICE%"=="layout" (
  set "LAYOUT_MODEL_DIR=%CD%\weights\layout"
  %PYTHON% -m uvicorn services.layout.main:app --host %INTERNAL_SERVICE_HOST% --port %LAYOUT_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="detection" (
  set "DET_MODEL_NAME="
  set "DET_MODEL_DIR="
  if "%DET_MODEL_VERSION%"=="" set "DET_MODEL_VERSION=v5"
  if "%MODEL_VARIANTS_CONFIG%"=="" set "MODEL_VARIANTS_CONFIG=%CD%\model_variants.json"
  %PYTHON% -m uvicorn services.text_det.main:app --host %INTERNAL_SERVICE_HOST% --port %DETECTION_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="rec-th" (
  set "REC_MODEL_NAME=th_PP-OCRv5_mobile_rec"
  set "REC_MODEL_VERSION=v5"
  set "REC_MODEL_DIR=%CD%\weights\recognition\rec-v5"
  %PYTHON% -m uvicorn services.text_rec.main:app --host %INTERNAL_SERVICE_HOST% --port %REC_TH_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="ocr-custom" (
  if "%REC_SERVICE_URL%"=="" set "REC_SERVICE_URL=http://127.0.0.1:%REC_TH_PORT%"
  %PYTHON% -m uvicorn services.ocr_pipeline_custom.main:app --host %INTERNAL_SERVICE_HOST% --port %OCR_CUSTOM_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="ocr-paddle" (
  set "DET_MODEL_NAME=PP-OCRv6_medium_det"
  set "DET_MODEL_DIR=%CD%\weights\detection\det-v6"
  set "REC_MODEL_NAME=th_PP-OCRv5_mobile_rec"
  set "REC_MODEL_DIR=%CD%\weights\recognition\rec-v5"
  %PYTHON% -m uvicorn services.ocr_pipeline_paddle.main:app --host %INTERNAL_SERVICE_HOST% --port %OCR_PADDLE_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="table-wired" (
  set "TABLE_MODEL_NAME=SLANeXt_wired"
  set "TABLE_MODEL_DIR=%CD%\weights\table-wired"
  %PYTHON% -m uvicorn services.table.main:app --host %INTERNAL_SERVICE_HOST% --port %TABLE_WIRED_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="table-wireless" (
  set "TABLE_MODEL_NAME=SLANeXt_wireless"
  set "TABLE_MODEL_DIR=%CD%\weights\table-wireless"
  %PYTHON% -m uvicorn services.table.main:app --host %INTERNAL_SERVICE_HOST% --port %TABLE_WIRELESS_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="table-v2" (
  if "%TABLE_V2_OCR_VERSION%"=="" set "TABLE_V2_OCR_VERSION=v5"
  if "%TABLE_V2_MODEL_CACHE_SIZE%"=="" set "TABLE_V2_MODEL_CACHE_SIZE=2"
  %PYTHON% -m uvicorn services.table_v2.main:app --host %INTERNAL_SERVICE_HOST% --port %TABLE_V2_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="siglip" (
  if "%SIGLIP_MODEL_NAME%"=="" set "SIGLIP_MODEL_NAME=google/siglip-so400m-patch14-384"
  set "SIGLIP_MODEL_DIR=%CD%\weights\siglip"
  %PYTHON% -m uvicorn services.siglip.main:app --host %INTERNAL_SERVICE_HOST% --port %SIGLIP_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="layout-pipeline" (
  if "%LAYOUT_SERVICE_URL%"=="" set "LAYOUT_SERVICE_URL=http://127.0.0.1:%LAYOUT_PORT%"
  %PYTHON% -m uvicorn services.layout_pipeline.main:app --host %INTERNAL_SERVICE_HOST% --port %LAYOUT_PIPELINE_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="table-pipeline" (
  if "%TABLE_WIRED_SERVICE_URL%"=="" set "TABLE_WIRED_SERVICE_URL=http://127.0.0.1:%TABLE_WIRED_PORT%"
  if "%TABLE_WIRELESS_SERVICE_URL%"=="" set "TABLE_WIRELESS_SERVICE_URL=http://127.0.0.1:%TABLE_WIRELESS_PORT%"
  if "%OCR_SERVICE_URL%"=="" set "OCR_SERVICE_URL=http://127.0.0.1:%OCR_CUSTOM_PORT%"
  %PYTHON% -m uvicorn services.table_pipeline.main:app --host %INTERNAL_SERVICE_HOST% --port %TABLE_PIPELINE_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="image-verification" (
  if "%SIGLIP_SERVICE_URL%"=="" set "SIGLIP_SERVICE_URL=http://127.0.0.1:%SIGLIP_PORT%"
  %PYTHON% -m uvicorn services.image_verification_pipeline.main:app --host %INTERNAL_SERVICE_HOST% --port %IMAGE_VERIFICATION_PORT% --workers 1
  goto :eof
)

if /I "%SERVICE%"=="gateway" (
  if "%LAYOUT_PIPELINE_URL%"=="" set "LAYOUT_PIPELINE_URL=http://127.0.0.1:%LAYOUT_PIPELINE_PORT%"
  if "%REC_SERVICE_URL%"=="" set "REC_SERVICE_URL=http://127.0.0.1:%REC_TH_PORT%"
  if "%OCR_CUSTOM_URL%"=="" set "OCR_CUSTOM_URL=http://127.0.0.1:%OCR_CUSTOM_PORT%"
  if "%OCR_PADDLE_URL%"=="" set "OCR_PADDLE_URL=http://127.0.0.1:%OCR_PADDLE_PORT%"
  if "%TABLE_PIPELINE_URL%"=="" set "TABLE_PIPELINE_URL=http://127.0.0.1:%TABLE_PIPELINE_PORT%"
  if "%TABLE_MODEL_URL%"=="" set "TABLE_MODEL_URL=http://127.0.0.1:%TABLE_V2_PORT%"
  if "%SIGLIP_URL%"=="" set "SIGLIP_URL=http://127.0.0.1:%SIGLIP_PORT%"
  if "%IMAGE_VERIFICATION_URL%"=="" set "IMAGE_VERIFICATION_URL=http://127.0.0.1:%IMAGE_VERIFICATION_PORT%"
  %PYTHON% -m uvicorn services.gateway.main:app --host %GATEWAY_HOST% --port %GATEWAY_PORT% --workers 1
  goto :eof
)

echo [ERROR] Unknown service: %SERVICE%
:usage
echo Usage: scripts\run-service.cmd ^<layout^|detection^|rec-th^|ocr-custom^|ocr-paddle^|table-wired^|table-wireless^|table-v2^|siglip^|layout-pipeline^|table-pipeline^|image-verification^|gateway^>
exit /b 1
