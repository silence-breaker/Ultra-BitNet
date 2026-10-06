@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0export_layer_bundle.ps1" %*
exit /b %ERRORLEVEL%
