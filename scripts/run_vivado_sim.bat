@echo off
setlocal
set ROOT=%~dp0..
call "G:\Vivado_2020.1\Vivado\2020.1\settings64.bat"
vivado -mode batch -source "%~dp0run_rtl_sim.tcl" -notrace
exit /b %ERRORLEVEL%
