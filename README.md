# SBS CamRemover

영상 속에 찍힌 **고정 거치카메라·삼각대**를 AI가 자동으로 지워주는 도구입니다.

카메라가 있던 자리를 주변 배경으로 자연스럽게 채워 넣어, 마치 처음부터 카메라가 없었던 것처럼 깨끗한 영상을 만들어 줍니다.

---

## 어떻게 작동하나요?

두 가지 AI 엔진을 제공합니다. 상황에 맞게 선택하세요.

### 1. MiniMax-Remover (클라우드 GPU 처리)

**원리**: 영상을 짧은 묶음(청크)으로 나눈 뒤, 각 묶음마다 AI가 "마스크 영역에 원래 어떤 배경이 있었을까?"를 추측해서 새로 그려 넣습니다.

내부적으로 **Diffusion(확산) 모델**을 사용합니다. 마스크 영역을 노이즈(잡음)로 가린 뒤, 주변 배경과 시간 흐름을 참고하여 잡음을 조금씩 제거해 나가면 자연스러운 배경이 복원됩니다. 이 과정을 클라우드 GPU 서버(RunPod)에서 처리하므로 내 컴퓨터에 GPU가 없어도 됩니다.

```
영상 업로드
    │
[1] 영상 분석        — 해상도, FPS, 총 프레임 수, 오디오 유무
[2] 마스크 전처리    — 제거 영역을 바이너리 마스크로 변환
[3] 청크 분할        — 45프레임씩 나눔 (앞뒤 9프레임 겹침)
[4] 서버 연결 확인
[5] 청크별 인페인팅  — 각 청크를 GPU 서버에 전송 → AI 추론
[6] 청크 합치기      — 겹침 구간을 부드럽게 이어붙임
[7] 오디오 복원      — 원본 소리를 결과 영상에 다시 합침
    │
결과 mp4 반환
```

- **장점**: 움직이는 피사체가 있어도, 복잡한 배경이어도 고품질 복원
- **필요 조건**: 클라우드 GPU 서버 연결

### 2. CamPatch (로컬 처리)

**원리**: 영상의 기준 프레임 1장에서 카메라 영역을 AI로 지운 "깨끗한 배경 이미지"를 만든 뒤, 이 배경을 모든 프레임에 붙여 넣습니다.

배경 생성에는 **LaMa**(Large Mask Inpainting) 모델을 사용합니다. LaMa는 큰 영역도 자연스럽게 채우는 이미지 인페인팅 AI로, 내 컴퓨터(로컬)에서 실행됩니다.

단, 배경을 단순히 덮어쓰면 카메라 앞을 지나가는 사람까지 사라집니다. 이를 방지하기 위해 **RVM**(RobustVideoMatting)이라는 AI로 "사람 영역"을 프레임마다 감지합니다. RVM이 사람이라고 판단한 부분은 원본을 유지하고, 나머지만 깨끗한 배경으로 교체합니다. **RVM은 GPU가 필요하므로 클라우드 서버에서 처리됩니다.**

```
영상 업로드
    │
[1] 영상 분석
[2] 마스크 전처리
[3] LaMa 인페인팅    — 기준 프레임 1장 → 깨끗한 배경 생성 (로컬)
[4] RVM 알파 추출    — GPU 서버에서 사람 영역 감지 (피사체 보호)
[5] 프레임별 블렌딩  — 사람 = 원본 유지, 나머지 = 깨끗한 배경 합성
[6] 오디오 복원
    │
결과 mp4 반환
```

- **장점**: 고정 카메라 영상에서 프레임 간 완벽한 일관성 (같은 배경이 매 프레임에 적용되므로 깜빡임이 없음)
- **필요 조건**: LaMa는 로컬 실행. **RVM 피사체 보호 사용 시 GPU 서버 연결 필요** (미사용 시 서버 없이도 동작)

### 어떤 엔진을 선택해야 하나요?

| 상황 | 추천 엔진 |
|------|-----------|
| 카메라가 고정이고, 배경이 단순하다 | **CamPatch** |
| 카메라 앞으로 사람이 자주 지나간다 | **CamPatch** (RVM 피사체 보호 ON) |
| 배경이 복잡하거나 움직이는 요소가 많다 | **MiniMax-Remover** |
| GPU 서버 없이 빠르게 처리하고 싶다 | **CamPatch** (RVM OFF) |

---

## 전체 구조

```
┌──────────────────────────────────────────┐    HTTP     ┌─────────────────────────────────┐
│           내 컴퓨터 (로컬)                │ ─────────▶ │     클라우드 GPU 서버 (RunPod)    │
│                                          │             │                                 │
│  Gradio 웹 UI                            │  /inpaint   │  MiniMax-Remover (배경 복원)     │
│    영상 업로드 → 마스크 그리기 → 엔진 선택 │  /rvm       │  RVM (사람 영역 감지)            │
│  CamRemoverAgent (파이프라인 관리)        │  /segment   │  SAM2 (클릭 → 마스크 자동 생성)  │
│  CamPatch (LaMa 배경 생성 + 블렌딩)      │ ◀───────── │                                 │
│                                          │  결과 반환   │                                 │
└──────────────────────────────────────────┘             └─────────────────────────────────┘
```

- **MiniMax-Remover**: GPU 서버에서 실행 (Diffusion 기반 영상 인페인팅)
- **LaMa**: 로컬에서 실행 (이미지 인페인팅으로 깨끗한 배경 1장 생성)
- **RVM**: GPU 서버에서 실행 (영상에서 사람 영역을 프레임별로 감지)
- **SAM2**: GPU 서버에서 실행 (클릭 한 번으로 제거 대상 자동 선택)

---

## 설치 방법

### Python을 모르시는 분 (권장)

아무것도 설치되어 있지 않아도 됩니다. `install.bat`이 모든 것을 자동으로 설치합니다.

**Windows:**

1. 이 프로젝트를 ZIP으로 다운로드하여 압축을 해제합니다.
2. 압축 해제한 폴더에서 `install.bat`을 **더블클릭**합니다.
3. 설치가 자동으로 진행됩니다 (최초 1회, 약 5~10분 소요):
   - Miniforge3 (경량 Python 환경 관리자)
   - Python 3.11 + 필요 패키지
   - ffmpeg (영상 처리용)
4. "설치 완료!" 메시지가 나오면 완료입니다.

**Mac / Linux:**

```bash
bash install.sh
```

### Python을 이미 사용 중인 분

```bash
pip install -r requirements.txt
```

> ffmpeg가 PATH에 있어야 합니다. 없으면 별도 설치하세요.

---

## 실행 방법

**Windows:** `run.bat` 더블클릭

**Mac / Linux:** `bash run.sh`

**직접 실행:**

```bash
python -m agent
```

실행 후 브라우저에서 `http://127.0.0.1:7860` 으로 접속합니다.

---

## 서버 연결

MiniMax-Remover, SAM2 클릭 마스킹, CamPatch RVM 피사체 보호 기능을 사용하려면 GPU 서버가 연결되어 있어야 합니다.

1. 화면 상단의 **"RunPod 연결 설정"** 을 펼칩니다.
2. **서버 URL** 칸에 공유받은 주소를 입력합니다 (예: `https://xxxxxx-8000.proxy.runpod.net`).
3. **"연결 테스트"** 를 클릭하여 "연결됨"을 확인합니다.

> 서버 URL은 관리자에게 문의하세요. 서버가 꺼져 있으면 연결되지 않습니다.

---

## 사용 방법

1. 영상 파일을 업로드합니다 (`.mp4` / `.mov` / `.avi`)
2. 마스크 모드를 선택합니다:
   - **SAM2 클릭**: 첫 프레임에서 제거할 카메라를 클릭 → AI가 자동으로 영역 감지 (서버 연결 필요)
   - **브러시**: 직접 브러시로 제거할 영역을 칠합니다
3. 인페인팅 엔진을 선택합니다:
   - **MiniMax-Remover**: GPU 서버에서 고품질 처리 (서버 연결 필요)
   - **CamPatch**: 로컬에서 처리 (RVM 피사체 보호 시에만 서버 필요)
4. **처리 시작**을 클릭합니다
5. 완료 후 결과 영상을 다운로드합니다

---

## 디렉토리 구조

```
camremover/
├── agent/                      # 로컬 파이프라인 코드
│   ├── ui.py                   # Gradio 웹 UI
│   ├── main.py                 # MiniMax 파이프라인 관리
│   ├── campatch.py             # CamPatch 엔진 (LaMa + RVM 블렌딩)
│   ├── segmenter.py            # SAM2 마스크 생성 (서버 호출)
│   ├── config.py               # 설정 로딩
│   ├── preprocessor.py         # 마스크 전처리, 영상 청크 분할
│   ├── runpod_client.py        # GPU 서버 통신
│   └── postprocessor.py        # 청크 합치기, 오디오 복원
│
├── docker/                     # GPU 서버 코드
│   ├── server.py               # FastAPI 추론 서버 (MiniMax + RVM + SAM2)
│   ├── setup.sh                # 서버 환경 셋업 (최초 1회)
│   └── requirements.txt        # 서버용 패키지
│
├── install.bat / install.sh    # 원클릭 설치 (권장)
├── setup.bat / setup.sh        # 간이 설치 (Python 사전 설치 필요)
├── run.bat / run.sh            # 실행 스크립트
├── config.yaml                 # 설정 파일 (git 제외)
├── config.example.yaml         # 설정 템플릿
├── deploy.sh                   # 서버 배포 스크립트
├── requirements.txt            # 로컬용 패키지
└── RUNPOD_RESTORE.md           # GPU 서버 복원 가이드
```

---

## 기술 스택

### 로컬 (내 컴퓨터)

| 구분 | 기술 |
|------|------|
| UI | Gradio |
| 인페인팅 | LaMa (simple-lama-inpainting) |
| 영상 처리 | OpenCV, FFmpeg |
| 언어 | Python 3.11+ |

### 클라우드 GPU 서버 (RunPod)

| 구분 | 기술 |
|------|------|
| 인페인팅 | MiniMax-Remover (Diffusion 기반 영상 복원) |
| 피사체 감지 | RVM — RobustVideoMatting (사람 영역 알파 매팅) |
| 마스크 생성 | SAM2 (클릭 기반 자동 세그멘테이션) |
| GPU | PyTorch 2.9, CUDA 12.9, NVIDIA RTX A5000 24GB |
| 서버 | FastAPI + Uvicorn |

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
    "total_mb": 24240.8,
    "gpu_name": "NVIDIA RTX A5000"
  }
}
```

### `POST /inpaint`

MiniMax-Remover 인페인팅. `multipart/form-data`로 영상 청크와 마스크를 전송합니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `video` | file | 비디오 청크 (.mp4) |
| `mask` | file | 바이너리 마스크 (.png) |
| `num_inference_steps` | int | 추론 스텝 수 (기본 6, 높을수록 고품질) |
| `seed` | int | 랜덤 시드 (기본 42) |

### `POST /rvm_matting`

RVM 피사체 감지. 영상에서 사람 영역의 알파 마스크를 추출합니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `video` | file | 입력 비디오 (.mp4) |
| `background` | file | 클린 배경 이미지 (.png) |
| `downsample_ratio` | float | 처리 해상도 비율 (기본 0.25) |

### `POST /segment`

SAM2 세그멘테이션. 클릭 좌표로 제거 대상의 마스크를 생성합니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `image` | file | RGB 이미지 (.png/.jpg) |
| `positive_points` | str | 제거 대상 좌표 `[[x,y],...]` |
| `negative_points` | str | 보존 대상 좌표 `[[x,y],...]` |

---

## Pod 복원

GPU 서버를 새로 생성해야 할 때는 [RUNPOD_RESTORE.md](RUNPOD_RESTORE.md)를 참고하세요.

---

## 문의

eight@sbs.co.kr
