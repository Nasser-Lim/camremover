#!/bin/bash
# deploy.sh — 로컬 변경사항을 GitHub에 push하고 Pod 서버를 업데이트한다.
#
# 사용법:
#   ./deploy.sh              # 현재 변경사항 커밋 + push + Pod 업데이트
#   ./deploy.sh "커밋 메시지"  # 커밋 메시지 지정
#
set -e

KEY="/c/Users/user/.ssh/id_ed25519"
REMOTE="xaku0flcejqm5v-64410fea@ssh.runpod.io"
REPO_DIR="/workspace/camremover"
SERVER_LOG="/tmp/server.log"

# ── 1) 로컬 커밋 + push ──────────────────────────────────────────────────
MSG="${1:-Update server}"

git add -A
if git diff --cached --quiet; then
    echo "변경사항 없음. push만 진행합니다."
else
    git commit -m "$MSG"
fi
git push origin master
echo "✓ GitHub push 완료"

# ── 2) Pod에서 git pull + 서버 재시작 ───────────────────────────────────
POD_CMD="
set -e
cd $REPO_DIR
git pull origin master
echo '✓ git pull 완료'

# 기존 서버 프로세스 종료
OLD_PID=\$(pgrep -f 'uvicorn server:app' | head -1)
if [ -n \"\$OLD_PID\" ]; then
    kill \$OLD_PID
    sleep 2
    echo \"✓ 기존 서버(PID \$OLD_PID) 종료\"
fi

# 서버 재시작
export PYTHONPATH=/workspace/pip_packages:\$PYTHONPATH
export MINIMAX_REMOVER_ROOT=/workspace/MiniMax-Remover
cd $REPO_DIR
nohup /usr/local/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 > $SERVER_LOG 2>&1 &
echo \"✓ 서버 시작 (PID \$!)\"
sleep 5
tail -5 $SERVER_LOG
"

echo "$POD_CMD" | ssh -o StrictHostKeyChecking=no -o RequestTTY=force -i "$KEY" "$REMOTE"
echo ""
echo "✓ 배포 완료"
