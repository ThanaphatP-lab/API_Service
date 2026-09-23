@echo off
setlocal EnableExtensions
set "ROOT=%~dp0.."
cd /d "%ROOT%"

if not exist ".venv-api312\Scripts\python.exe" (
  echo [ERROR] API virtual environment not found. Run scripts\setup-local.cmd first.
  exit /b 1
)

if "%DEMO_HOST%"=="" set "DEMO_HOST=127.0.0.1"
if "%DEMO_PORT%"=="" set "DEMO_PORT=8501"
if "%GATEWAY_PORT%"=="" set "GATEWAY_PORT=8080"
if "%GATEWAY_URL%"=="" set "GATEWAY_URL=http://127.0.0.1:%GATEWAY_PORT%"
if "%TABLE_V2_PORT%"=="" set "TABLE_V2_PORT=8013"
if "%TABLE_V2_URL%"=="" set "TABLE_V2_URL=http://127.0.0.1:%TABLE_V2_PORT%"
.venv-api312\Scripts\python.exe -m streamlit run demo\app.py --server.address %DEMO_HOST% --server.port %DEMO_PORT%
