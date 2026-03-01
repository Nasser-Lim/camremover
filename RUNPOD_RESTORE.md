# RunPod Pod 복원 가이드 (MiniMax-Remover)

## 인프라 구조

```
Network Volume (camremover-vol, 20GB, EU-SE-1) — ID: opzuu8b9hu
  └─ /workspace/
       ├─ MiniMax-Remover/      ← 모델 소스코드
       ├─ weights/minimax-remover/  ← 가중치 (~2.76GB)
       ├─ pip_packages/         ← 모든 pip 패키지 (~2.4GB, sam2 포함)
       └─ camremover/server.py  ← FastAPI 서버
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
# RunPod SSH 프록시로 접속 (웹 터미널 활성화 필요)
ssh xaku0flcejqm5v-64410fea@ssh.runpod.io -i ~/.ssh/id_ed25519

# [Python 버전 불일치 시] 누락 패키지 보완 설치 (Python 3.12용 빌드)
pip install safetensors regex tokenizers sentencepiece \
    uvicorn[standard] fastapi python-multipart \
    -q --target /workspace/pip_packages

# 서버 시작
cd /workspace/camremover
PYTHONPATH=/workspace/pip_packages:$PYTHONPATH \
MINIMAX_REMOVER_ROOT=/workspace/MiniMax-Remover \
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

### 2단계: 로컬에서 setup.sh 전송 후 실행

```bash
# 로컬에서 파일 전송
scp -P <PORT> docker/server.py docker/setup.sh root@<IP>:/tmp/

# Pod SSH 접속
ssh -p <PORT> root@<IP>

# setup.sh 실행 (소스 clone + 가중치 다운로드 + pip 설치 + 서버 시작)
bash /tmp/setup.sh
```

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

```bash
ssh -p <PORT> root@<IP>

# 기존 프로세스 확인 후 종료
PID=$(ss -tlnp | grep 8000 | grep -oP "pid=\K[0-9]+")
kill -9 $PID

# 재시작
cd /workspace/camremover
PYTHONPATH=/workspace/pip_packages:$PYTHONPATH \
MINIMAX_REMOVER_ROOT=/workspace/MiniMax-Remover \
nohup /usr/local/bin/python -m uvicorn server:app \
    --host 0.0.0.0 --port 8000 --workers 1 \
    > /tmp/server.log 2>&1 &

sleep 25 && tail -20 /tmp/server.log
```

---

## server.py 업데이트 (로컬 수정 반영)

```bash
scp -P <PORT> docker/server.py root@<IP>:/workspace/camremover/server.py
# 서버 재시작 (위 "서버 재시작" 섹션 참고)
```
