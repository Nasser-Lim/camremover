#!/bin/bash
set -e

ENV_NAME="camremover"
MINIFORGE_DIR="$HOME/miniforge3"
CONDA_EXE="$MINIFORGE_DIR/bin/conda"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================="
echo " CamRemover 설치 (Miniforge 기반)"
echo " Python / conda 사전 설치 불필요"
echo "============================================="
echo

# ── Step 1: Miniforge ──────────────────────────
if [ -f "$CONDA_EXE" ]; then
    echo "[OK] Miniforge: $MINIFORGE_DIR"
else
    echo "Miniforge3 다운로드 중..."
    ARCH=$(uname -m)
    OS=$(uname -s)
    if [ "$OS" = "Darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then
            URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
        else
            URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
        fi
    else
        URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    fi

    curl -L -o /tmp/Miniforge3.sh "$URL"
    bash /tmp/Miniforge3.sh -b -p "$MINIFORGE_DIR"
    rm /tmp/Miniforge3.sh
    echo "[OK] Miniforge3 설치 완료"
fi
echo

# ── Step 2: conda 초기화 ───────────────────────
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

# ── Step 3: camremover 환경 생성 ───────────────
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[OK] conda 환경 '$ENV_NAME' 이미 존재"
else
    echo "conda 환경 생성 중 (Python 3.11)..."
    conda create -n "$ENV_NAME" python=3.11 -y -q
    echo "[OK] 환경 생성 완료"
fi
conda activate "$ENV_NAME"
echo

# ── Step 4: Python 패키지 ──────────────────────
echo "PyTorch 설치 중 (CPU 버전, ~300MB)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q

echo "패키지 설치 중..."
pip install gradio numpy opencv-python Pillow requests pyyaml scipy tqdm sam2 simple-lama-inpainting -q
echo "[OK] Python 패키지 완료"
echo

# ── Step 5: ffmpeg ─────────────────────────────
if command -v ffmpeg &>/dev/null; then
    echo "[OK] ffmpeg: 시스템에 있음"
else
    echo "ffmpeg 설치 중 (conda-forge)..."
    conda install -c conda-forge ffmpeg -y -q
    echo "[OK] ffmpeg 설치 완료"
fi
echo

# ── 설치 정보 저장 ─────────────────────────────
echo "$MINIFORGE_DIR" > "$SCRIPT_DIR/.conda_root.txt"
echo "$ENV_NAME"      > "$SCRIPT_DIR/.conda_env.txt"

echo "============================================="
echo " 설치 완료!"
echo "============================================="
echo
echo "  ./run.sh 를 실행하면 앱이 시작됩니다."
echo "  실행 후 브라우저에서 http://127.0.0.1:7860 접속"
echo "  상단 'RunPod 연결 설정'에서 서버 주소 입력"
echo
