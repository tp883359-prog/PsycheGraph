@echo off
rem One-click launcher for the PsycheGraph demo (container path A).
rem Double-click this file, or run it from CMD / PowerShell.
rem Extra arguments are forwarded to scripts\demo_up.ps1
rem (for example: demo_up.cmd -NoBrowser -SkipPrewarm).
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\demo_up.ps1" %*
if errorlevel 1 (
    echo.
    echo [!] Startup failed - the message above says why.
    pause
)
endlocal
