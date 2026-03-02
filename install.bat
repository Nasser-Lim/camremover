@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

set CONDA_ROOT=%USERPROFILE%\miniforge3
set ENV_NAME=camremover
set CONDA_EXE=%CONDA_ROOT%\Scripts\conda.exe
set SCRIPT_DIR=%~dp0

echo =============================================
echo  CamRemover 설치 (Miniforge 기반)
echo  Python / conda 사전 설치 불필요
echo =============================================
echo.

REM ── Step 1: Miniforge ──────────────────────
if exist "%CONDA_EXE%" (
    echo [OK] Miniforge: %CONDA_ROOT%
) else (
    echo Miniforge3 다운로드 중 (약 100MB^) ...
    curl -L --progress-bar -o "%TEMP%\Miniforge3.exe" ^
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
    if errorlevel 1 (
        echo.
        echo [오류] 다운로드 실패. 인터넷 연결을 확인하세요.
        pause & exit /b 1
    )
    echo Miniforge3 설치 중 (관리자 권한 불필요^) ...
    "%TEMP%\Miniforge3.exe" /S /D=%CONDA_ROOT%
    del "%TEMP%\Miniforge3.exe" 2>nul
    if not exist "%CONDA_EXE%" (
        echo [오류] Miniforge 설치 실패.
        pause & exit /b 1
    )
    echo [OK] Miniforge3 설치 완료
)
echo.

REM ── Step 2: conda 환경 활성화 ─────────────
call "%CONDA_ROOT%\Scripts\activate.bat" base
if errorlevel 1 (
    echo [오류] conda 초기화 실패.
    pause & exit /b 1
)

REM ── Step 3: camremover 환경 생성 ──────────
conda env list 2>nul | findstr /B "%ENV_NAME% " >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] conda 환경 '%ENV_NAME%' 이미 존재
) else (
    echo conda 환경 생성 중 (Python 3.11^) ...
    conda create -n %ENV_NAME% python=3.11 -y -q
    if errorlevel 1 (
        echo [오류] 환경 생성 실패.
        pause & exit /b 1
    )
    echo [OK] 환경 생성 완료
)
call conda activate %ENV_NAME%
echo.

REM ── Step 4: Python 패키지 ─────────────────
echo PyTorch 설치 중 (CPU 버전, ~300MB^) ...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
if errorlevel 1 ( echo [오류] PyTorch 설치 실패. & pause & exit /b 1 )

echo 패키지 설치 중 ...
pip install gradio numpy opencv-python Pillow requests pyyaml scipy tqdm sam2 simple-lama-inpainting -q
if errorlevel 1 ( echo [오류] 패키지 설치 실패. & pause & exit /b 1 )
echo [OK] Python 패키지 완료
echo.

REM ── Step 5: ffmpeg ────────────────────────
set FFMPEG_DIR=%SCRIPT_DIR%tools\ffmpeg
set FFMPEG_EXE=%FFMPEG_DIR%\ffmpeg.exe

if exist "%FFMPEG_EXE%" (
    echo [OK] ffmpeg: %FFMPEG_DIR%
    goto ffmpeg_done
)

where ffmpeg >nul 2>&1
if %errorlevel% == 0 (
    echo [OK] ffmpeg: 시스템 PATH에 있음
    goto ffmpeg_done
)

echo ffmpeg 다운로드 중 (~80MB^) ...
curl -L --progress-bar -o "%TEMP%\ffmpeg.zip" ^
  "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
if errorlevel 1 (
    echo [경고] ffmpeg 자동 다운로드 실패.
    echo         수동 설치: https://ffmpeg.org/download.html
    goto ffmpeg_done
)

echo ffmpeg 압축 해제 중 ...
mkdir "%FFMPEG_DIR%" 2>nul
powershell -NoProfile -Command ^
  "Expand-Archive -Path '%TEMP%\ffmpeg.zip' -DestinationPath '%TEMP%\ffmpeg_tmp' -Force"

REM bin\ 안의 exe 두 개만 복사
for /d %%D in ("%TEMP%\ffmpeg_tmp\ffmpeg-*") do (
    copy /y "%%D\bin\ffmpeg.exe"  "%FFMPEG_DIR%\" >nul 2>&1
    copy /y "%%D\bin\ffprobe.exe" "%FFMPEG_DIR%\" >nul 2>&1
)
del "%TEMP%\ffmpeg.zip" 2>nul
rmdir /s /q "%TEMP%\ffmpeg_tmp" 2>nul

if exist "%FFMPEG_EXE%" (
    echo [OK] ffmpeg 설치 완료: %FFMPEG_DIR%
) else (
    echo [경고] ffmpeg 설치 실패. 수동 설치: https://ffmpeg.org/download.html
)

:ffmpeg_done
echo.

REM ── 설치 정보 저장 ────────────────────────
echo %CONDA_ROOT%> "%SCRIPT_DIR%.conda_root.txt"
echo %ENV_NAME%>   "%SCRIPT_DIR%.conda_env.txt"

echo =============================================
echo  설치 완료!
echo =============================================
echo.
echo  run.bat 을 더블클릭하면 앱이 실행됩니다.
echo  실행 후 브라우저에서 http://127.0.0.1:7860 접속
echo  상단 "RunPod 연결 설정"에서 서버 주소 입력
echo.
pause
