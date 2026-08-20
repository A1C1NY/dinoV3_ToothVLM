@echo off
setlocal
set "TOOTH_VLM_ROOT=%~dp0..\.."
python "%TOOTH_VLM_ROOT%\app\scripts\launch.py"
exit /b %errorlevel%
