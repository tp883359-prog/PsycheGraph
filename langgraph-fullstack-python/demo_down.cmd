@echo off
rem One-click stop for the PsycheGraph demo stack.
rem Double-click this file, or run it from CMD / PowerShell.
rem Add -WipeVolumes to also delete the ~4.5 GB Hugging Face weights volume.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\demo_down.ps1" %*
if errorlevel 1 (
    echo.
    echo [!] Stop failed - the message above says why.
    pause
)
endlocal
