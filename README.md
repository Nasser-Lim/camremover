# SBS CamRemover

영상 속 고정 거치카메라 · 삼각대를 AI로 자동 제거하는 에이전트.

두 가지 인페인팅 엔진을 지원한다:
- **MiniMax-Remover** (GPU 서버): DiT 기반 Diffusion — 고품질, RunPod Pod 필요
- **CamPatch** (로컬): LaMa 이미지 인페인팅 + RVM 알파 블렌딩 — 고정 카메라에 완벽한 시간적 일관성, GPU 서버 불필요

---

## 아키텍처

```
┌─────────────────────────────────────────────────┐      HTTP (multipart)     ┌────────────────────────────────────┐
│              로컬 (Windows/Mac)                  │ ─────────────────────── ▶ │       RunPod GPU Pod               │
│                                                 │                           │                                    │
│  Gradio UI (agent/ui.py)                        │   POST /inpaint           │  FastAPI (docker/server.py)        │
│    ↓ 영상 업로드 + 마스크 그리기 + 엔진 선택     │   video chunk + mask PNG  │    MiniMaxRemoverModel             │
│  CamRemoverAgent (agent/main.py)                │ ◀ ─────────────────────── │      AutoencoderKLWan (VAE)        │
│    └─ MiniMax 모드: 파이프라인 오케스트레이션    │   inpainted mp4 bytes     │      Transformer3DModel (DiT)      │
│  CamPatch (agent/campatch.py)                   │                           │      FlowMatchEulerDiscreteScheduler│
│    └─ LaMa 클린 레퍼런스 생성                   │   POST /rvm_matting       │  RobustVideoMatting (RVM)          │
│    └─ RVM 알파 추출 (Pod 호출)                  │ ◀ ─────────────────────── │      mobilenetv3 (torch.hub)       │
│    └─ 프레임별 알파 블렌딩                       │   alpha mp4 bytes         │  SAM2 (세그멘테이션)               │
│  Segmenter (agent/segmenter.py)                 │                           │      sam2.1-hiera-tiny             │
│    └─ SAM2 클릭 → 마스크 생성 (Pod 호출)         │   POST /segment           │                                    │
└─────────────────────────────────────────────────┘                           └────────────────────────────────────┘
```

---

## 인페인팅 엔진

### MiniMax-Remover (GPU 서버)

RunPod Pod에서 실행되는 DiT 기반 Diffusion 인페인팅.

**파이프라인 (7단계):**
```
영상 업로드
    │
[1] 영상 분석        — 해상도, FPS, 총 프레임 수, 오디오 유무
[2] 마스크 전처리    — RGBA → 바이너리, 리사이즈, dilation(팽창)
[3] 청크 분할        — 45프레임씩 9프레임 겹침으로 분할 (마지막 청크 패딩)
[4] Pod 연결 확인    — /health 폴링 (최대 300초)
[5] 청크별 인페인팅  — 각 청크를 Pod에 전송 → MiniMax-Remover 추론
[6] 청크 합치기      — 겹침 구간 크로스페이드 블렌딩 (패딩 프레임 제거)
[7] 오디오 복원      — 원본 오디오를 ffmpeg로 재합성
    │
결과 mp4 반환
```

### CamPatch (로컬)

LaMa 이미지 인페인팅으로 클린 배경 1장을 생성한 뒤, 전 프레임에 패치를 블렌딩.
고정 카메라 영상에서 완벽한 시간적 일관성을 제공하며, GPU 서버 없이 로컬 CPU/GPU에서 동작.

**파이프라인 (6단계):**
```
영상 업로드
    │
[1] 영상 분석
[2] 마스크 전처리
[3] LaMa 인페인팅    — 기준 프레임 1장으로 클린 레퍼런스 배경 생성
[4] RVM 알파 추출    — 마스크 bbox 크롭 영상을 Pod에 전송
                       RobustVideoMatting으로 피사체 알파 마스크 추출
                       (RVM 워밍업: 첫 프레임 15회 반복 → 배경 수렴 후 제거)
[5] 프레임별 블렌딩  — original × rvm_pha + clean_ref × (1 - rvm_pha)
                       마스크 밖: 원본 그대로 / 경계: feather 블렌딩
[6] 오디오 복원
    │
결과 mp4 반환
```

---

## 디렉토리 구조

```
camremover/
├── agent/                      # 로컬 오케스트레이션 (Python 패키지)
│   ├── ui.py                   # Gradio 웹 UI (엔진 선택, SAM2 클릭 마스킹)
│   ├── main.py                 # CamRemoverAgent — MiniMax 파이프라인
│   ├── campatch.py             # CamPatch 엔진 — LaMa + RVM 블렌딩
│   ├── segmenter.py            # SAM2 세그멘테이션 (Pod 호출)
│   ├── config.py               # 설정 로딩 및 검증 (dataclass)
│   ├── preprocessor.py         # 마스크 전처리, 영상 청크 분할
│   ├── runpod_client.py        # RunPod Pod HTTP 클라이언트
│   └── postprocessor.py        # 청크 합치기, 오디오 복원
│
├── docker/                     # RunPod Pod 서버 코드
│   ├── server.py               # FastAPI 추론 서버 (MiniMax + RVM + SAM2)
│   ├── setup.sh                # Pod 환경 셋업 스크립트 (최초 1회)
│   └── requirements.txt        # Pod용 Python 패키지
│
├── tests/                      # 단위 테스트
│   ├── test_preprocessor.py
│   └── test_postprocessor.py
│
├── install.bat / install.sh    # 원클릭 설치 (Miniforge 기반, 권장)
├── setup.bat / setup.sh        # 간이 설치 (Python 사전 설치 필요)
├── run.bat / run.sh            # 실행 스크립트
├── config.yaml                 # 전체 설정 파일 (git 제외)
├── config.example.yaml         # 배포용 설정 템플릿
├── deploy.sh                   # 로컬 → GitHub push + Pod 자동 배포
├── requirements.txt            # 로컬용 Python 패키지
└── RUNPOD_RESTORE.md           # Pod 복원 가이드
```

---

## 기술 스택

### 로컬 (오케스트레이션)

| 구분 | 기술 |
|------|------|
| UI | Gradio (Blocks API, ImageEditor, gr.File) |
| 인페인팅 (로컬) | simple-lama-inpainting (LaMa) |
| 영상 처리 | OpenCV, FFmpeg |
| HTTP 통신 | requests (Session, multipart/form-data) |
| 설정 | PyYAML + Python dataclass |
| 언어 | Python 3.11+ |

### RunPod Pod (GPU 추론)

| 구분 | 기술 |
|------|------|
| 서버 | FastAPI + Uvicorn |
| 인페인팅 | MiniMax-Remover (VAE + Transformer3DModel + FlowMatchEulerDiscreteScheduler) |
| 알파 매팅 | RobustVideoMatting (mobilenetv3, torch.hub) |
| 세그멘테이션 | SAM2 (sam2.1-hiera-tiny) |
| GPU | PyTorch 2.9.0+cu129, CUDA 12.9 |
| 이미지 | `runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204` (Python 3.12) |
| 영상 처리 | OpenCV, imageio |
| 동시성 | asyncio.Lock (추론별 직렬화) |
| 모델 가중치 | MiniMax ~2.76GB, RVM ~50MB, SAM2 ~155MB |

### 인프라

| 구분 | 기술 |
|------|------|
| GPU 서버 | RunPod Pod (RTX A5000 24GB) |
| 스토리지 | RunPod Network Volume 20GB (`camremover-vol`, EU-SE-1) |
| 네트워킹 | RunPod Proxy (`{pod_id}-8000.proxy.runpod.net`) |

---

## 주요 설계 결정

### 1. 이중 엔진 구조
- **MiniMax-Remover**: 피사체가 움직이거나 마스크 영역에 복잡한 배경 복원이 필요한 경우
- **CamPatch**: 고정 카메라 영상에서 완벽한 일관성이 필요한 경우. LaMa로 배경 1장을 생성하고 전 프레임에 적용하므로 청크 경계 아티팩트가 없음

### 2. CamPatch RVM 블렌딩
클린 레퍼런스를 단순히 붙이면 마스크 영역의 피사체(사람 등)가 사라진다. RVM으로 피사체 알파를 추출해 `original × pha + clean_ref × (1-pha)` 공식으로 합성. 마스크 bbox 크롭만 Pod에 전송해 네트워크 부하 최소화.

### 3. RVM 워밍업 프레임
RVM은 recurrent 모델이라 초반 프레임에서 배경 수렴이 불안정해 플리커링 발생. 첫 프레임을 15번 반복해서 영상 앞에 붙여 RVM을 사전 수렴시키고, 결과에서 해당 프레임을 제거.

### 4. MiniMax-Remover 청크 처리
모델 학습 프레임 수는 81이지만, 타임아웃 방지를 위해 기본 45프레임씩 처리 (config에서 변경 가능). 마지막 청크가 부족하면 마지막 프레임으로 패딩 후 결과에서 제거. 청크 경계는 9프레임 크로스페이드 블렌딩.

### 5. 해상도 다운스케일 전략
원본이 `max_inpaint_resolution`(기본 480px 높이)을 초과하면 다운스케일 후 처리 → 업스케일 + 마스크 feather 블렌딩으로 합성.

### 6. gr.File 입력 (Windows PermissionError 우회)
`gr.Video`는 Windows에서 `.mov` 업로드 시 임시 파일 잠금으로 PermissionError 발생. `gr.File`로 변경 후 처리 시점에만 ffmpeg로 변환.

### 7. torch 버전 관리
Pod 이미지(CUDA 12.9)와 pip_packages의 torch 버전이 다르면 `CUBLAS_STATUS_INVALID_VALUE` 발생. `/workspace/pip_packages`의 torch는 반드시 `2.9.0+cu129`여야 함.

---

## 설치 및 실행

### 사전 요구사항

없음. `install.bat`이 Python, 패키지, ffmpeg를 모두 자동 설치한다.

> Python이 이미 설치되어 있다면 `setup.bat` / `setup.sh` 로 가상환경만 구성할 수도 있다 (ffmpeg는 별도 설치 필요).

### 설치 방법 (Windows)

1. 이 프로젝트를 ZIP으로 다운로드하여 압축을 해제한다.
2. 압축 해제한 폴더에서 `install.bat`을 **더블클릭**한다.
3. 설치가 자동으로 진행된다 (최초 1회, 약 5~10분 소요):
   - Miniforge3 (경량 Python 환경 관리자) 설치
   - Python 3.11 + 필요 패키지 설치
   - ffmpeg (영상 처리용) 자동 다운로드
4. "설치 완료!" 메시지가 나오면 아무 키나 눌러 닫는다.

### 설치 방법 (Mac / Linux)

```bash
bash install.sh
```

### 실행

**Windows:** `run.bat` 더블클릭

**Mac / Linux:** `bash run.sh`

실행 후 브라우저에서 `http://127.0.0.1:7860` 으로 접속한다.

### 서버 연결

1. 화면 상단의 **"RunPod 연결 설정"** 을 펼친다.
2. **서버 URL** 칸에 공유받은 주소를 입력한다 (예: `https://xxxxxx-8000.proxy.runpod.net`).
3. **"연결 테스트"** 를 클릭하여 "연결됨"을 확인한다.

> 서버 URL은 관리자에게 문의. 서버가 꺼져 있으면 연결되지 않는다.

### 사용 방법

1. 영상 파일 업로드 (`.mp4` / `.mov` / `.avi`)
2. 마스크 모드 선택:
   - **SAM2 클릭**: 첫 프레임에서 제거할 카메라를 클릭 → AI가 자동으로 영역 감지 (서버 연결 필요)
   - **브러시**: 직접 브러시로 영역을 칠함
3. 인페인팅 엔진 선택:
   - **MiniMax-Remover**: GPU 서버에서 처리. 고품질, 서버 연결 필요
   - **CamPatch**: 로컬에서 처리. GPU 서버 불필요 (RVM 피사체 보호 시에만 서버 필요)
4. **처리 시작** 클릭
5. 완료 후 결과 영상 다운로드

---

## Pod 서버 API

### `GET /health`

```json
{
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": true,
  "engine": "minimax-remover",
  "gpu_memory": {
    "allocated_mb": 2708.0,
    "reserved_mb": 3010.0,
    "total_mb": 24240.8,
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
| `num_inference_steps` | int | 디퓨전 추론 스텝 수 (기본 6) |
| `seed` | int | 랜덤 시드 (기본 42) |
| `mask_dilation` | int | 마스크 팽창 반복 (기본 6) |
| `feather_px` | int | 경계 블렌딩 반경 (기본 5) |

**응답**: 인페인팅된 mp4 바이트 (`application/octet-stream`)

### `POST /rvm_matting`

**요청** (`multipart/form-data`):

| 필드 | 타입 | 설명 |
|------|------|------|
| `video` | file | 입력 비디오 (.mp4) |
| `background` | file | 클린 레퍼런스 배경 이미지 (.png) |
| `downsample_ratio` | float | RVM 처리 해상도 비율 (기본 0.25) |

**응답**: 알파 그레이스케일 mp4 바이트 (0=배경, 255=전경)

### `POST /segment`

**요청** (`multipart/form-data`):

| 필드 | 타입 | 설명 |
|------|------|------|
| `image` | file | RGB 이미지 (.png/.jpg) |
| `positive_points` | str | 제거 대상 좌표 JSON `[[x,y],...]` |
| `negative_points` | str | 보존 대상 좌표 JSON `[[x,y],...]` |

**응답**: PNG 바이너리 마스크 (255=제거, 0=유지)

### `POST /unload_model` / `POST /reload_model`

GPU 메모리 수동 관리용. `target`: `"minimax"` | `"rvm"` | `"sam2"` | `"all"`

---

## 테스트

```bash
pytest tests/ -v
```

---

## Pod 복원

GPU Pod를 새로 생성해야 할 때는 [RUNPOD_RESTORE.md](RUNPOD_RESTORE.md)를 참고한다.

---

## 문의

eight@sbs.co.kr
