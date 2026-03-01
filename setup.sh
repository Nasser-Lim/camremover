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

echo
echo "============================================"
echo " 설치 완료!"
echo "============================================"
echo
echo "./run.sh 를 실행하세요."
echo "앱 실행 후 상단 'RunPod 연결 설정'에서 서버 주소를 입력하면 됩니다."
echo
