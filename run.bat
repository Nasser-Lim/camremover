@echo off
setlocal

set SCRIPT_DIR=%~dp0

REM ── conda 환경 방식 (install.bat 사용 시) ──
if exist "%SCRIPT_DIR%.conda_root.txt" (
    set /p CONDA_ROOT=<"%SCRIPT_DIR%.conda_root.txt"
    set /p ENV_NAME=<"%SCRIPT_DIR%.conda_env.txt"

    REM 로컬 ffmpeg 우선
    if exist "%SCRIPT_DIR%tools\ffmpeg\ffmpeg.exe" (
        set PATH=%SCRIPT_DIR%tools\ffmpeg;%PATH%
    )

    call "%CONDA_ROOT%\Scripts\activate.bat" base
    call conda activate %ENV_NAME%
    python -m agent
    exit /b
)

REM ── venv 방식 (setup.bat 사용 시) ──────────
if exist "%SCRIPT_DIR%.venv" (
    call "%SCRIPT_DIR%.venv\Scripts\activate.bat"
    python -m agent
    exit /b
)

echo [오류] 설치가 되어있지 않습니다.
echo        install.bat (권장) 또는 setup.bat 을 먼저 실행하세요.
pause
