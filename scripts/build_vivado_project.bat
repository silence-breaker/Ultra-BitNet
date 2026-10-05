@echo off
setlocal
call "G:\Vivado_2020.1\Vivado\2020.1\settings64.bat"
vivado -mode batch -source "%~dp0build_vivado_project.tcl" -tclargs %*
exit /b %ERRORLEVEL%
