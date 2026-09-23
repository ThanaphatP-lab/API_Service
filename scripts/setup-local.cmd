@echo off
setlocal EnableExtensions
set "ROOT=%~dp0.."
cd /d "%ROOT%"
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=all"

rem Paddle and PyTorch ship different cuDNN DLL builds on Windows. Keep them in
rem separate virtual environments to prevent WinError 127 during DLL loading.
if /I "%TARGET%"=="paddle" goto :setup_paddle
if /I "%TARGET%"=="siglip" goto :setup_siglip
if /I "%TARGET%"=="api" goto :setup_api
if /I not "%TARGET%"=="all" goto :usage

:setup_all
call :setup_env .venv-paddle312 requirements-paddle.txt
if errorlevel 1 exit /b 1
call :setup_env .venv-siglip312 requirements-siglip.txt
if errorlevel 1 exit /b 1
call :setup_env .venv-api312 requirements-api.txt
if errorlevel 1 exit /b 1
goto :complete

:setup_paddle
call :setup_env .venv-paddle312 requirements-paddle.txt
if errorlevel 1 exit /b 1
goto :complete

:setup_siglip
call :setup_env .venv-siglip312 requirements-siglip.txt
if errorlevel 1 exit /b 1
goto :complete

:setup_api
call :setup_env .venv-api312 requirements-api.txt
if errorlevel 1 exit /b 1
goto :complete

:complete
echo.
echo [OK] Requested local environment setup is complete: %TARGET%
echo.
echo Start a profile with: scripts\model-stack.cmd start core-stack
echo Start TableV2 with: scripts\model-stack.cmd start table-v2-stack
echo Start the demo with: scripts\model-stack.cmd start demo
exit /b 0

:usage
echo Usage: scripts\setup-local.cmd ^<all^|paddle^|siglip^|api^>
exit /b 1

:setup_env
set "ENV_DIR=%~1"
set "REQ_FILE=%~2"

if not exist "%ENV_DIR%\Scripts\python.exe" (
  echo [INFO] Creating %ENV_DIR%...
  py -3.12 -m venv "%ENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Python 3.12 could not create %ENV_DIR%.
    echo Install Python 3.12 x64 from python.org, then run this script again.
    exit /b 1
  )
)

"%ENV_DIR%\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 (
  echo [INFO] Repairing %ENV_DIR% after a Python update...
  py -3.12 -m venv --upgrade "%ENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] %ENV_DIR% is broken and could not be repaired.
    echo Rename or remove that environment, then run this script again.
    exit /b 1
  )
)

echo [INFO] Installing %REQ_FILE% into %ENV_DIR%...
"%ENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%ENV_DIR%\Scripts\python.exe" -m pip install -r "%REQ_FILE%"
if errorlevel 1 exit /b 1
exit /b 0
