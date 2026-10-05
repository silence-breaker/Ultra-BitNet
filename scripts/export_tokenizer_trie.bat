@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0export_tokenizer_trie.ps1" %*
exit /b %ERRORLEVEL%
