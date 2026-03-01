@echo off
setlocal
echo ============================================
echo  CamRemover Setup
echo ============================================
echo.

REM Python 버전 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo https://www.python.org/downloads/ 에서 Python 3.10 이상을 설치하세요.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Python %PYVER% 감지됨

REM 가상환경 생성
if exist .venv (
    echo [스킵] .venv 이미 존재합니다.
) else (
    echo 가상환경 생성 중...
    python -m venv .venv
)

REM 가상환경 활성화
call .venv\Scripts\activate.bat

REM pip 업그레이드
echo pip 업그레이드 중...
python -m pip install --upgrade pip --quiet

REM PyTorch CPU 설치 (CUDA 없이도 동작, 용량 절약)
echo PyTorch 설치 중 (CPU 버전, ~300MB)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

REM 나머지 패키지 설치
echo 나머지 패키지 설치 중...
pip install gradio numpy opencv-python Pillow requests pyyaml scipy tqdm sam2 simple-lama-inpainting --quiet

REM config.yaml 생성
if exist config.yaml (
    echo [스킵] config.yaml 이미 존재합니다.
) else (
    copy config.example.yaml config.yaml >nul
    echo config.yaml 생성됨 — server_url을 채워주세요!
)

echo.
echo ============================================
echo  설치 완료!
echo ============================================
echo.
echo 다음 단계:
echo   1. config.yaml 을 열어 custom_url 에 서버 주소 입력
echo   2. run.bat 실행
echo.
pause
