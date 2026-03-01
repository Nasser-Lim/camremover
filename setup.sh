#!/bin/bash
set -e

echo "============================================"
echo " CamRemover Setup"
echo "============================================"
echo

# Python 확인
if ! command -v python3 &>/dev/null; then
    echo "[오류] python3가 설치되어 있지 않습니다."
    echo "https://www.python.org/downloads/ 에서 Python 3.10 이상을 설치하세요."
    exit 1
fi

PYVER=$(python3 --version)
echo "$PYVER 감지됨"

# 가상환경 생성
if [ -d ".venv" ]; then
    echo "[스킵] .venv 이미 존재합니다."
else
    echo "가상환경 생성 중..."
    python3 -m venv .venv
fi

# 가상환경 활성화
source .venv/bin/activate

# pip 업그레이드
echo "pip 업그레이드 중..."
pip install --upgrade pip --quiet

# PyTorch CPU
echo "PyTorch 설치 중 (CPU 버전, ~300MB)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

# 나머지 패키지
echo "나머지 패키지 설치 중..."
pip install gradio numpy opencv-python Pillow requests pyyaml scipy tqdm sam2 simple-lama-inpainting --quiet

# config.yaml 생성
if [ -f "config.yaml" ]; then
    echo "[스킵] config.yaml 이미 존재합니다."
else
    cp config.example.yaml config.yaml
    echo "config.yaml 생성됨 — server_url을 채워주세요!"
fi

echo
echo "============================================"
echo " 설치 완료!"
echo "============================================"
echo
echo "다음 단계:"
echo "  1. config.yaml 을 열어 custom_url 에 서버 주소 입력"
echo "  2. ./run.sh 실행"
echo
