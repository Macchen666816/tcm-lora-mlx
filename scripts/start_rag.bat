@echo off
REM Quick start: TCM RAG microservice (8090) + RAG admin console (8091)
REM Double-click this file, or run it from cmd.
REM NOTE: keep this file ASCII-only so cmd does not garble it.

setlocal
set PY=D:\conda_\envs\lora\python.exe
set ROOT=%~dp0..

if not exist "%PY%" (
  echo [ERROR] Python not found: %PY%
  echo         Edit PY in this script to point at your conda env lora python.
  pause
  exit /b 1
)

REM The local proxy 127.0.0.1:18081 is often dead; clear it for child processes.
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set ALL_PROXY=
set all_proxy=
REM Load the embedding model from local cache instead of the network.
set HF_HUB_OFFLINE=1

pushd "%ROOT%"

echo.
echo Starting RAG microservice on http://127.0.0.1:8090 ...
start "TCM RAG Service 8090" cmd /k ""%PY%" -m rag_service.server --host 0.0.0.0 --port 8090"

timeout /t 2 /nobreak >nul

echo Starting RAG admin console on http://127.0.0.1:8091 ...
start "TCM RAG Console 8091" cmd /k ""%PY%" rag_console\server.py --port 8091"

popd

echo.
echo Both windows are opening. The service needs ~20-30s to load the model.
echo   RAG service : http://127.0.0.1:8090/health
echo   RAG console : http://127.0.0.1:8091
echo.
echo LAN access (for the teammate MacBook) uses this machine's IP:
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /c:"IPv4"') do @echo    %%i
echo.
echo Reminder: allow inbound TCP 8090 in the firewall once:
echo   New-NetFirewallRule -DisplayName "TCM RAG 8090" -Direction Inbound -LocalPort 8090 -Protocol TCP -Action Allow
echo.
echo To stop both services, run stop_rag.bat
pause
