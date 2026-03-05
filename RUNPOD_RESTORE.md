# RunPod Pod 복원 가이드 (MiniMax-Remover)

## 인프라 구조

```
Network Volume (camremover-vol, 20GB, EU-SE-1) — ID: opzuu8b9hu
  └─ /workspace/
       ├─ MiniMax-Remover/      ← 모델 소스코드
       ├─ weights/minimax-remover/  ← 가중치 (~2.76GB)
       ├─ pip_packages/         ← 모든 pip 패키지 (~2.4GB, sam2 포함)
       ├─ torch_cache/          ← torch.hub 캐시 (RVM 모델 소스+가중치)
       └─ camremover/           ← GitHub 레포 clone
            ├─ docker/server.py ← 소스 (배포 시 server.py로 복사)
            ├─ server.py        ← uvicorn 실행 대상 (deploy.sh가 복사)
            ├─ agent/           ← 로컬 UI 코드
            └─ deploy.sh        ← 배포 스크립트
```

- Pod를 삭제해도 Volume의 모든 데이터가 유지됨
- **새 Pod에 Volume만 연결하면 pip install 없이 바로 서버 시작 가능**

---

## 시나리오 A: Network Volume에 데이터가 있을 때 (일반적)

### 1단계: Pod 생성 (Network Volume 연결)

**GraphQL API로 생성 (권장):**

```bash
curl -X POST https://api.runpod.io/graphql \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"query":"mutation { podFindAndDeployOnDemand(input: { name: \"camremover\", imageName: \"runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204\", gpuTypeId: \"NVIDIA RTX A5000\", gpuCount: 1, containerDiskInGb: 20, networkVolumeId: \"opzuu8b9hu\", volumeMountPath: \"/workspace\", ports: \"8000/http,22/tcp\", cloudType: SECURE, env: [{ key: \"MINIMAX_REMOVER_ROOT\", value: \"/workspace/MiniMax-Remover\" }, { key: \"PUBLIC_KEY\", value: \"<SSH_PUBLIC_KEY>\" }] }) { id runtime { ports { ip isIpPublic privatePort publicPort type } } } }"}'
```

> `volumeMountPath: "/workspace"` 반드시 명시 (누락 시 mount target empty 에러)
> 인증 헤더: `Authorization: Bearer <API_KEY>` (`api-key` 헤더 아님)

**웹 UI로 생성 시:**

| 항목 | 값 |
|------|-----|
| GPU | RTX A5000 24GB 이상 권장 |
| Image | `runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204` |
| Container Disk | 20GB |
| **Network Volume** | **`camremover-vol` 선택** |
| Mount Path | `/workspace` |
| Expose Ports | `8000/http`, `22/tcp` |
| 환경변수 | `MINIMAX_REMOVER_ROOT=/workspace/MiniMax-Remover` |

### 2단계: 서버 시작

> **주의**: 이미지가 `1.0.3-cu1290-torch290-ubuntu2204` (Python 3.12) 기반이면,
> `/workspace/pip_packages`가 Python 3.10용으로 빌드된 경우 C 확장 패키지가 누락될 수 있음.
> 아래 보완 설치 명령어를 먼저 실행할 것.

```bash
# 직접 TCP SSH 접속 (portMappings에서 22번 포트 확인)
ssh -p <TCP_PORT> root@<PUBLIC_IP> -i ~/.ssh/id_ed25519

# [Python 버전 불일치 시] 누락 패키지 보완 설치 (Python 3.12용 빌드)
pip install safetensors regex tokenizers sentencepiece \
    uvicorn[standard] fastapi python-multipart \
    -q --target /workspace/pip_packages

# docker/server.py → server.py 복사 (GitHub 배포 방식)
cp /workspace/camremover/docker/server.py /workspace/camremover/server.py

# 서버 시작
cd /workspace/camremover
PYTHONPATH=/workspace/pip_packages:$PYTHONPATH \
MINIMAX_REMOVER_ROOT=/workspace/MiniMax-Remover \
TORCH_HOME=/workspace/torch_cache \
nohup /usr/local/bin/python -m uvicorn server:app \
    --host 0.0.0.0 --port 8000 --workers 1 \
    > /tmp/server.log 2>&1 &

sleep 30 && tail -30 /tmp/server.log
```

> `/workspace/pip_packages`에 모든 패키지(sam2, diffusers 등) 영구 저장됨

### 3단계: 헬스체크

```bash
curl https://<POD_ID>-8000.proxy.runpod.net/health
```

정상 응답:
```json
{"status": "healthy", "model_loaded": true, "engine": "minimax-remover"}
```

### 4단계: 로컬 config.yaml에서 Pod ID 변경

```yaml
runpod:
  pod_id: "새로운Pod_ID"   # ← 여기만 변경
```

---

## 시나리오 B: Network Volume이 비어있을 때 (최초 설정)

### 1단계: Pod 생성

시나리오 A의 1단계와 동일.

### 2단계: GitHub에서 clone 후 setup.sh 실행

```bash
# 직접 TCP SSH 접속
ssh -p <TCP_PORT> root@<PUBLIC_IP> -i ~/.ssh/id_ed25519

# GitHub 인증 설정 (최초 1회)
git config --global credential.helper store
echo 'https://x-access-token:<GH_TOKEN>@github.com' > ~/.git-credentials
chmod 600 ~/.git-credentials

# 레포 clone
cd /workspace
git clone https://github.com/Nasser-Lim/camremover.git

# setup.sh 실행 (소스 clone + 가중치 다운로드 + pip 설치 + 서버 시작)
bash /workspace/camremover/docker/setup.sh
```

> `GH_TOKEN`: 로컬에서 `gh auth token`으로 확인

`setup.sh`가 자동으로 수행하는 작업:
1. apt 패키지 설치 (git, ffmpeg 등)
2. MiniMax-Remover git clone → `/workspace/MiniMax-Remover`
3. HuggingFace에서 모델 가중치 다운로드 (~2.76GB) → `/workspace/weights/minimax-remover`
4. pip 패키지 설치 → `/workspace/pip_packages`
5. 서버 코드 배포 → `/workspace/camremover/server.py`
6. FastAPI 서버 시작 (포트 8000)

소요 시간: 약 5~10분 (가중치 다운로드 포함)

---

## 서버 재시작 (필요 시)

**로컬에서 deploy.sh 사용 (권장):**

```bash
bash deploy.sh
```

**Pod에 직접 접속해서 재시작:**

```bash
ssh -p <TCP_PORT> root@<PUBLIC_IP> -i ~/.ssh/id_ed25519

# 기존 프로세스 종료
OLD_PID=$(pgrep -f 'uvicorn server:app' | head -1)
[ -n "$OLD_PID" ] && kill $OLD_PID && sleep 2

# 재시작
cd /workspace/camremover
PYTHONPATH=/workspace/pip_packages:$PYTHONPATH \
MINIMAX_REMOVER_ROOT=/workspace/MiniMax-Remover \
TORCH_HOME=/workspace/torch_cache \
nohup /usr/local/bin/python -m uvicorn server:app \
    --host 0.0.0.0 --port 8000 --workers 1 \
    > /tmp/server.log 2>&1 &

sleep 25 && tail -20 /tmp/server.log
```

---

## server.py 업데이트 — GitHub 배포 (권장)

GitHub 레포: `https://github.com/Nasser-Lim/camremover` (private)

### 일상적인 배포 (코드 수정 후)

```bash
# 로컬에서 한 명령으로 push + Pod 자동 업데이트
bash deploy.sh "변경 내용 설명"
```

`deploy.sh`가 자동으로 수행하는 작업:
1. `git add -A && git commit && git push origin master`
2. Pod SSH 접속 → `git pull origin master`
3. `docker/server.py` → `/workspace/camremover/server.py` 복사
4. 기존 uvicorn 프로세스 종료 후 재시작

### 새 Pod에 최초 설정 (Pod 재생성 시)

```bash
# 1) Pod SSH 접속
ssh -p <TCP_PORT> root@<PUBLIC_IP> -i ~/.ssh/id_ed25519

# 2) GitHub 인증 설정 (최초 1회)
git config --global credential.helper store
echo 'https://x-access-token:<GH_TOKEN>@github.com' > ~/.git-credentials
chmod 600 ~/.git-credentials

# 3) 레포 clone
cd /workspace
git clone https://github.com/Nasser-Lim/camremover.git
```

> **GH_TOKEN**: `gh auth token`으로 로컬에서 확인
> 이후에는 `bash deploy.sh`만 실행하면 자동 배포됨

### Pod 재시작 후 서버 시작 (deploy.sh 이용)

```bash
# 변경사항 없어도 서버만 재시작하려면
bash deploy.sh
```

> 변경사항이 없으면 커밋은 건너뛰고 Pod pull + 재시작만 수행함

### torch.hub 캐시 (RVM 모델)

RVM(RobustVideoMatting) 모델은 최초 `/rvm_matting` 요청 시 자동 다운로드됨.
`TORCH_HOME=/workspace/torch_cache`로 설정되어 있으므로 Pod 재시작 후에도 재다운로드 없이 즉시 사용 가능.

---

## 로컬 Gradio UI 실행

### 전제조건

- Python 3.11+ 설치
- 의존성 설치: `pip install -r requirements.txt`
- ffmpeg PATH에 있거나 `tools/ffmpeg/` 아래 위치

### 실행

```bash
# 프로젝트 루트에서
python -m agent
```

접속 주소: `http://127.0.0.1:7860`

### config.yaml 설정

```yaml
runpod:
  pod_id: "3ydaps81iyucx1"   # 현재 Pod ID
```

또는 UI의 **RunPod 연결 설정** 아코디언에서 서버 URL 직접 입력:

```
https://3ydaps81iyucx1-8000.proxy.runpod.net
```

### install.bat / run.bat 사용 시 (비기술 사용자)

```bat
install.bat   ← 최초 1회 (Miniforge + 패키지 + ffmpeg 자동 설치)
run.bat       ← 이후 매번 실행
```
