@echo off
setlocal
if not defined VIVADO_SETTINGS (echo Set VIVADO_SETTINGS to the tool settings64.bat path before running this script. & exit /b 2)
call "%VIVADO_SETTINGS%"
vivado_hls -f "%~dp0build_attention_hls.tcl"
if errorlevel 1 exit /b %ERRORLEVEL%
if not exist "%~dp0..\build\hls_rtl\bitnet_attention_axis\bitnet_attention_axis.v" exit /b 1
exit /b 0
