@echo off
setlocal
if not defined VITIS_SETTINGS (echo Set VITIS_SETTINGS to the tool settings64.bat path before running this script. & exit /b 2)
call "%VITIS_SETTINGS%"
xsct "%~dp0run_on_board.tcl" %*
exit /b %ERRORLEVEL%
