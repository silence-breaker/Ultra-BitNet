@echo off
setlocal
if not defined VIVADO_SETTINGS (echo Set VIVADO_SETTINGS to the tool settings64.bat path before running this script. & exit /b 2)
call "%VIVADO_SETTINGS%"
vivado -mode batch -source "%~dp0build_vivado_project.tcl" -tclargs %*
exit /b %ERRORLEVEL%
