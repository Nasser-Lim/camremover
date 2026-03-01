# CamRemover

고정 거치카메라를 영상에서 자동으로 제거하는 AI 에이전트.

사용자가 첫 프레임에서 카메라 영역을 브러시로 칠하면, MiniMax-Remover (DiT 기반 Diffusion) AI가 해당 영역을 자연스럽게 복원한다.

---

## 아키텍처

```
┌─────────────────────────────────────┐      HTTP (multipart)     ┌─────────────────────────────────┐
│         로컬 (Windows/Mac)           │ ─────────────────────── ▶ │     RunPod GPU Pod               │
│                                     │                           │                                 │
│  Gradio UI (agent/ui.py)            │   POST /inpaint           │  FastAPI (docker/server.py)     │
│    ↓ 영상 업로드 + 마스크 그리기     │   video chunk + mask PNG  │    ↓                            │
│  CamRemoverAgent (agent/main.py)    │ ◀ ─────────────────────── │  MiniMaxRemoverModel            │
│    ↓ 파이프라인 오케스트레이션       │   inpainted mp4 bytes     │    AutoencoderKLWan (VAE)       │
│  Preprocessor → RunPodClient        │                           │    Transformer3DModel (DiT)     │
│  → Postprocessor                    │                           │    UniPCMultistepScheduler      │
└─────────────────────────────────────┘                           └─────────────────────────────────┘
```

### 처리 파이프라인 (7단계)

```
영상 업로드
    │
    ▼
[1] 영상 분석        — 해상도, FPS, 총 프레임 수, 오디오 유무
    │
    ▼
[2] 마스크 전처리    — RGBA → 바이너리, 리사이즈, dilation(팽창)
    │
    ▼
[3] 청크 분할        — 81프레임씩 11프레임 겹침으로 분할 (마지막 청크 패딩)
    │
    ▼
[4] Pod 연결 확인    — /health 폴링 (최대 300초)
    │
    ▼
[5] 청크별 인페인팅  — 각 청크를 Pod에 전송 → MiniMax-Remover 추론
    │
    ▼
[6] 청크 합치기      — 겹침 구간 크로스페이드 블렌딩 (패딩 프레임 제거)
    │
    ▼
[7] 오디오 복원      — 원본 오디오를 ffmpeg로 재합성
    │
    ▼
결과 mp4 반환
```

---

## 디렉토리 구조

```
camremover/
├── agent/                      # 로컬 오케스트레이션 (Python 패키지)
│   ├── ui.py                   # Gradio 웹 UI
│   ├── main.py                 # CamRemoverAgent — 파이프라인 오케스트레이터
│   ├── config.py               # 설정 로딩 및 검증 (dataclass)
│   ├── preprocessor.py         # 마스크 전처리, 영상 청크 분할
│   ├── runpod_client.py        # RunPod Pod HTTP 클라이언트
│   └── postprocessor.py        # 청크 합치기, 오디오 복원
│
├── docker/                     # RunPod Pod 서버 코드
│   ├── server.py               # FastAPI 추론 서버 (MiniMax-Remover)
│   ├── setup.sh                # Pod 환경 셋업 스크립트 (원클릭)
│   └── requirements.txt        # Pod용 Python 패키지
│
├── tests/                      # 단위 테스트
│   ├── test_preprocessor.py
│   └── test_postprocessor.py
│
├── config.yaml                 # 전체 설정 파일
├── requirements.txt            # 로컬용 Python 패키지
└── RUNPOD_RESTORE.md           # Pod 복원 가이드
```

---

## 기술 스택

### 로컬 (오케스트레이션)

| 구분 | 기술 |
|------|------|
| UI | Gradio 4.x (Blocks API, ImageEditor, gr.File) |
| 영상 처리 | OpenCV, FFmpeg (subprocess pipe) |
| HTTP 통신 | requests (Session, multipart/form-data) |
| 설정 | PyYAML + Python dataclass |
| 언어 | Python 3.12 |

### RunPod Pod (GPU 추론)

| 구분 | 기술 |
|------|------|
| 서버 | FastAPI + Uvicorn |
| AI 모델 | MiniMax-Remover (DiT 기반 Diffusion — VAE + Transformer3DModel + UniPCMultistepScheduler) |
| GPU | PyTorch 2.1+, CUDA 12.1 |
| 영상 처리 | OpenCV, imageio, FFmpeg |
| 동시성 | asyncio.Lock (GPU 단일 직렬화) |
| 모델 가중치 | ~2.76GB (HuggingFace: zibojia/minimax-remover) |
| 언어 | Python 3.10 |

### 인프라

| 구분 | 기술 |
|------|------|
| GPU 서버 | RunPod Pod (RTX A5000 24GB 이상 권장) |
| 네트워킹 | RunPod Proxy (`{pod_id}-8000.proxy.runpod.net`) |

---

## 주요 설계 결정

### 1. MiniMax-Remover (DiT 기반 Diffusion)
GAN 기반 ProPainter에서 DiT 기반 MiniMax-Remover로 교체. 고정 거치카메라는 "다른 프레임에서 빌려오기"가 불가능하여 GAN의 100% 생성에 의존 → 블러/뭉개짐 발생. MiniMax-Remover는 문맥 기반 생성으로 품질이 우수하다.

### 2. 청크 기반 처리 + 크로스페이드 블렌딩
긴 영상을 81프레임(MiniMax-Remover 학습 프레임 수)씩 나눠서 처리한다. 마지막 청크가 81프레임 미만이면 마지막 프레임을 복제하여 패딩한 뒤, 결과에서 패딩 프레임을 제거한다. 청크 경계의 이음새를 11프레임의 크로스페이드로 자연스럽게 연결한다.

### 3. 해상도 다운스케일 전략
GPU VRAM 한계 대응. 원본이 `max_inpaint_resolution`(기본 480px 높이)을 초과하면:
- 처리: 다운스케일 → MiniMax-Remover 추론
- 복원: 업스케일 → 원본 비마스크 영역과 feather 블렌딩 합성

### 4. gr.File 입력 (Windows PermissionError 우회)
`gr.Video`는 Windows에서 `.mov` 업로드 시 임시 파일 잠금으로 PermissionError 발생. `gr.File`로 변경 후 처리 시점에만 ffmpeg로 변환.

### 5. 재시도 로직
네트워크 오류/일시적 서버 오류에 지수 백오프 재시도 (5초, 10초, 20초, 최대 3회).

---

## 로컬 실행

### 설치

```bash
pip install -r requirements.txt
```

ffmpeg가 PATH에 있어야 한다.

### 설정

`config.yaml`에서 Pod ID와 처리 옵션을 설정한다:

```yaml
runpod:
  pod_id: "your-pod-id"       # RunPod Pod ID
  timeout_seconds: 600

video:
  max_inpaint_resolution: 480 # 처리 높이 (px), 낮을수록 빠름
  chunk_size: 81              # 청크당 프레임 수 (MiniMax-Remover 기본값)
  chunk_overlap: 11           # 청크 간 겹침 프레임

minimax_remover:
  num_inference_steps: 12     # 디퓨전 스텝 (6~12, 높을수록 고품질)
  seed: 42                    # 재현성용 시드
  mask_dilation: 6            # 마스크 팽창 반복 횟수

mask:
  dilation_kernel_size: 7     # 마스크 팽창 크기 (px)
  feather_radius: 5           # 경계 블렌딩 반경
```

### 실행

```bash
python -m agent.ui
# 접속: http://127.0.0.1:7860
```

### 사용 방법

1. 영상 파일 업로드 (`.mp4` / `.mov` / `.avi`)
2. 첫 프레임에서 빨간 브러시로 제거할 카메라 영역을 칠한다
3. Pod ID 확인 후 **처리 시작** 클릭
4. 완료 후 결과 영상을 다운로드

---

## Pod 서버 API

Pod가 실행 중일 때 직접 호출 가능.

### `GET /health`

```json
{
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": true,
  "engine": "minimax-remover",
  "gpu_memory": {
    "allocated_mb": 4200.0,
    "reserved_mb": 5000.0,
    "total_mb": 24576.0,
    "gpu_name": "NVIDIA RTX A5000"
  }
}
```

### `POST /inpaint`

**요청** (`multipart/form-data`):

| 필드 | 타입 | 설명 |
|------|------|------|
| `video` | file | 비디오 청크 (.mp4) |
| `mask` | file | 바이너리 마스크 (.png) |
| `max_inpaint_height` | int | 처리 최대 높이 (기본 480) |
| `num_inference_steps` | int | 디퓨전 추론 스텝 수 (기본 12, 범위 6~12) |
| `seed` | int | 재현성용 랜덤 시드 (기본 42) |
| `mask_dilation` | int | 마스크 팽창 반복 횟수 (기본 6) |
| `feather_px` | int | 경계 블렌딩 반경 (기본 5) |

**응답**: 인페인팅된 mp4 바이트 (`application/octet-stream`)

---

## 테스트

```bash
pytest tests/ -v
```

---

## Pod 복원

GPU Pod를 새로 생성해야 할 때는 [RUNPOD_RESTORE.md](RUNPOD_RESTORE.md)를 참고한다.


## runpod api key(GraphQL용):
REDACTED