@echo off
REM Stop the TCM RAG microservice (8090) and the RAG admin console (8091).
REM ASCII-only file.

setlocal enabledelayedexpansion

for %%P in (8090 8091) do (
  set FOUND=0
  for /f "tokens=5" %%A in ('netstat -ano ^| findstr ":%%P" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%A >nul 2>&1
    if !errorlevel! equ 0 (
      echo Stopped PID %%A on port %%P
      set FOUND=1
    )
  )
  if !FOUND! equ 0 echo No listener on port %%P
)

echo Done.
pause
