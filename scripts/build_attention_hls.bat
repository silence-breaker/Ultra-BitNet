@echo off
setlocal
call "G:\Vivado_2020.1\Vivado\2020.1\settings64.bat"
vivado_hls -f "%~dp0build_attention_hls.tcl"
if errorlevel 1 exit /b %ERRORLEVEL%
if not exist "%~dp0..\build\hls_rtl\bitnet_attention_axis\bitnet_attention_axis.v" exit /b 1
exit /b 0
