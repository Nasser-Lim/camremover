@echo off
if not exist .venv (
    echo [오류] .venv가 없습니다. setup.bat을 먼저 실행하세요.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python -m agent
