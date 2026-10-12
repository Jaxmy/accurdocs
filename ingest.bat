@echo off
REM accurdocs document ingest
REM Usage: ingest.bat "path\to\file.pdf" [doc_id] [version] [effective_date]
REM Example: ingest.bat "docs\policy.pdf" policy-2024 2024-03-01 2024-03-01
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d %~dp0
set PY=%~dp0.venv312\Scripts\python.exe
if not exist "%PY%" set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
if "%~1"=="" (
  echo Usage: ingest.bat "file path" [doc_id] [version] [effective_date]
  exit /b 1
)
set ARGS=ingest "%~1"
if not "%~2"=="" set ARGS=%ARGS% --doc-id %~2
if not "%~3"=="" set ARGS=%ARGS% --version %~3
if not "%~4"=="" set ARGS=%ARGS% --effective-date %~4
"%PY%" scripts\query.py %ARGS%
