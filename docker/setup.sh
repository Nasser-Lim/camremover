#!/bin/bash
# CamRemover — RunPod Pod 셋업 스크립트 (MiniMax-Remover)
# 사용법: bash setup.sh
# Network Volume: /workspace (RunPod 기본 마운트)

set -e  # 에러 발생 시 즉시 중단

# pip와 동일한 Python 인터프리터 사용 (runpod 이미지는 pip=python3.12, python3=python3.10 불일치)
PYTHON="$(pip show pip 2>/dev/null | grep Location | head -1 | sed 's|.*/lib/\(python[0-9.]*\)/.*|/usr/local/bin/\1|' || echo python3)"
if ! command -v "$PYTHON" &>/dev/null; then PYTHON=/usr/local/bin/python; fi
PIP="$PYTHON -m pip"

WORKSPACE="/workspace"
MINIMAX_DIR="$WORKSPACE/MiniMax-Remover"
WEIGHTS_DIR="$WORKSPACE/weights/minimax-remover"
SERVER_DIR="$WORKSPACE/camremover"

echo "========================================"
echo " CamRemover Setup (MiniMax-Remover)"
echo "========================================"

# ── 1) 시스템 패키지 ──
echo "[1/6] 시스템 패키지 설치..."
apt-get update -qq && apt-get install -y -qq \
    git \
    ffmpeg \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    > /dev/null
echo "      완료"

# ── 2) MiniMax-Remover 클론 (없으면) ──
echo "[2/6] MiniMax-Remover 준비..."
if [ ! -d "$MINIMAX_DIR" ]; then
    git clone --depth=1 https://github.com/zibojia/MiniMax-Remover.git "$MINIMAX_DIR"
    echo "      클론 완료"
else
    echo "      이미 존재 — 스킵"
fi

# ── 3) 모델 가중치 다운로드 (없으면) ──
echo "[3/6] 모델 가중치 다운로드..."
if [ ! -d "$WEIGHTS_DIR/transformer" ]; then
    echo "      HuggingFace에서 다운로드 중 (~2.76GB)..."
    $PIP install -q huggingface_hub
    $PYTHON -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zibojia/minimax-remover', local_dir='$WEIGHTS_DIR')"
    echo "      다운로드 완료"
else
    echo "      이미 존재 — 스킵"
fi

# ── 4) Python 패키지 설치 ──
echo "[4/6] Python 패키지 설치..."
$PIP install -q \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    numpy \
    opencv-python-headless \
    Pillow \
    scipy \
    imageio \
    imageio-ffmpeg \
    einops \
    tqdm \
    pyyaml \
    diffusers \
    decord \
    accelerate \
    transformers \
    huggingface_hub \
    requests
echo "      완료"

# ── 5) 서버 코드 배포 ──
echo "[5/6] 서버 코드 배포..."
mkdir -p "$SERVER_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/server.py" "$SERVER_DIR/server.py"

# 환경변수 설정
export MINIMAX_REMOVER_ROOT="$MINIMAX_DIR"
echo "export MINIMAX_REMOVER_ROOT=$MINIMAX_DIR" >> ~/.bashrc

echo "      완료"

# ── 6) 서버 시작 ──
echo "[6/6] FastAPI 서버 시작..."
echo ""
echo "========================================"
echo " 서버 정보"
echo "   주소: http://0.0.0.0:8000"
echo "   헬스체크: GET /health"
echo "   인페인팅: POST /inpaint"
echo "========================================"
echo ""

cd "$SERVER_DIR"
MINIMAX_REMOVER_ROOT="$MINIMAX_DIR" \
    $PYTHON -m uvicorn server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
