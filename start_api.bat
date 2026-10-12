@echo off
REM accurdocs REST API + GUI server
REM GUI: http://127.0.0.1:%ACCURDOCS_PORT%   Swagger: http://127.0.0.1:%ACCURDOCS_PORT%/docs
REM Config: .env in this directory (LLM key / url / model) — never committed
title accurdocs API server
cd /d %~dp0
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if "%ACCURDOCS_PORT%"=="" set ACCURDOCS_PORT=8000

REM ---- pick python: local venv first, else system ----
set PY=%~dp0.venv312\Scripts\python.exe
if not exist "%PY%" set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

REM ---- port check: a flash-exit usually means another server holds the port ----
netstat -ano | findstr /R /C:":%ACCURDOCS_PORT% .*LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo [ERROR] Port %ACCURDOCS_PORT% is already in use - probably another accurdocs server is running.
    echo Set ACCURDOCS_PORT to use a different port.
    echo.
    pause
    exit /b 1
)

echo Starting accurdocs on http://127.0.0.1:%ACCURDOCS_PORT% ...
"%PY%" -m uvicorn policyrag.adapters.rest.app:app --host 127.0.0.1 --port %ACCURDOCS_PORT% --workers 1 --limit-concurrency 32

echo.
echo Server stopped (exit code %errorlevel%). Press any key to close.
pause >nul
