@echo off
setlocal
call "G:\Vivado_2020.1\Vitis\2020.1\settings64.bat"
xsct "%~dp0run_layer_chain_on_board.tcl" %*
exit /b %ERRORLEVEL%
