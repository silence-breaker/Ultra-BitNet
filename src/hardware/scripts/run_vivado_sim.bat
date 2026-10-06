@echo off
setlocal
set ROOT=%~dp0..
if not defined VIVADO_SETTINGS (echo Set VIVADO_SETTINGS to the tool settings64.bat path before running this script. & exit /b 2)
call "%VIVADO_SETTINGS%"
vivado -mode batch -source "%~dp0run_rtl_sim.tcl" -notrace
exit /b %ERRORLEVEL%
