@echo off
REM accurdocs MCP server (stdio transport)
REM Mount into Claude Code / Cursor / VS Code / any MCP client
REM Config: .env in this directory (LLM key / url / model) — never committed
title accurdocs MCP server
cd /d %~dp0
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PY=%~dp0.venv312\Scripts\python.exe
if not exist "%PY%" set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=python
"%PY%" -m policyrag.adapters.mcp.server %*
REM pause only on failure (normal exit must return to the MCP client immediately)
if %errorlevel% neq 0 (
    echo.
    echo MCP server failed (exit code %errorlevel%). Press any key to close.
    pause >nul
)
