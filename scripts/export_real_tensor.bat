@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0export_real_tensor.ps1" %*
exit /b %ERRORLEVEL%
