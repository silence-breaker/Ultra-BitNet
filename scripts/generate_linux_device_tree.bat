@echo off
setlocal
if not defined PROCESSOR_ARCHITECTURE set "PROCESSOR_ARCHITECTURE=AMD64"
call "G:\Vivado_2020.1\Vitis\2020.1\settings64.bat"
xsct "%~dp0generate_linux_device_tree.tcl"
exit /b %ERRORLEVEL%
