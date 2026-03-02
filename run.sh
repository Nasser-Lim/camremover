#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── conda 환경 방식 (install.sh 사용 시) ──
if [ -f "$SCRIPT_DIR/.conda_root.txt" ]; then
    CONDA_ROOT=$(cat "$SCRIPT_DIR/.conda_root.txt")
    ENV_NAME=$(cat "$SCRIPT_DIR/.conda_env.txt")
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    python -m agent
    exit 0
fi

# ── venv 방식 (setup.sh 사용 시) ────────────
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    python -m agent
    exit 0
fi

echo "[오류] 설치가 되어있지 않습니다."
echo "       install.sh (권장) 또는 setup.sh 를 먼저 실행하세요."
exit 1
