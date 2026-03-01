#!/bin/bash
if [ ! -d ".venv" ]; then
    echo "[오류] .venv가 없습니다. setup.sh를 먼저 실행하세요."
    exit 1
fi
source .venv/bin/activate
python -m agent
