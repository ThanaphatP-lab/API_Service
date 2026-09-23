@echo off
setlocal EnableExtensions

rem Windows entry point matching model-stack.sh command syntax.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0model-stack.ps1" %*
exit /b %errorlevel%
