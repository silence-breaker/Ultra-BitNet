@echo off
setlocal
if not defined PROCESSOR_ARCHITECTURE set "PROCESSOR_ARCHITECTURE=AMD64"
if not defined VITIS_SETTINGS (echo Set VITIS_SETTINGS to the tool settings64.bat path before running this script. & exit /b 2)
call "%VITIS_SETTINGS%"
xsct "%~dp0generate_linux_device_tree.tcl"
exit /b %ERRORLEVEL%
