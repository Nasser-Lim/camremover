"""
FastAPI 추론 서버 — RunPod GPU Pod 내부에서 실행.
MiniMax-Remover (DiT 기반 비디오 인페인팅) + SAM2 (세그멘테이션).

엔드포인트:
  GET  /health   — 서버/GPU 상태 확인
  POST /inpaint  — 비디오 청크 + 마스크 → 인페인팅된 mp4
  POST /segment  — 이미지 + 포인트 → 세그멘테이션 마스크 PNG
"""

import asyncio
import gc
import io
import logging
import os
import sys
import tempfile
import time

import cv2
import imageio
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

# MiniMax-Remover 소스코드를 sys.path에 추가
MINIMAX_ROOT = os.environ.get("MINIMAX_REMOVER_ROOT", "/workspace/MiniMax-Remover")
if MINIMAX_ROOT not in sys.path:
    sys.path.insert(0, MINIMAX_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("camremover-server")

app = FastAPI(title="CamRemover Inference Server")


class MiniMaxRemoverModel:
    """MiniMax-Remover 모델 래퍼."""

    def __init__(self, weights_dir: str, device: str = "cuda"):
        from diffusers.models import AutoencoderKLWan
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformer_minimax_remover import Transformer3DModel
        from pipeline_minimax_remover import Minimax_Remover_Pipeline

        self.device = torch.device(device)

        logger.info("Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(weights_dir, "vae"),
            torch_dtype=torch.float16,
        )

        logger.info("Loading Transformer...")
        transformer = Transformer3DModel.from_pretrained(
            os.path.join(weights_dir, "transformer"),
            torch_dtype=torch.float16,
        )

        logger.info("Loading Scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(weights_dir, "scheduler"),
        )

        logger.info("Building pipeline...")
        self.pipe = Minimax_Remover_Pipeline(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        ).to(self.device)

    def inpaint(
        self,
        frames: list,
        mask: np.ndarray,
        num_inference_steps: int = 12,
        seed: int = 42,
        mask_dilation: int = 6,
    ) -> list:
        """
        비디오 프레임의 마스크 영역을 인페인팅한다.

        Args:
            frames: List[HxWx3 uint8 BGR] — 비디오 프레임
            mask: HxW uint8 (255=제거, 0=유지) — 정적 마스크
            num_inference_steps: 디퓨전 스텝 수 (6~12)
            seed: 재현성용 랜덤 시드
            mask_dilation: 마스크 팽창 반복 횟수

        Returns:
            List[HxWx3 uint8 BGR] — 인페인팅된 프레임
        """
        num_frames = len(frames)
        h, w = frames[0].shape[:2]

        # 1) BGR [0,255] → RGB [-1, 1] → torch tensor
        images_np = np.stack([
            cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
            for f in frames
        ])  # (T, H, W, 3)
        images = torch.from_numpy(images_np).to(dtype=torch.float32)

        # 2) mask [0,255] → [0, 1] → torch tensor (T, H, W, 1)
        #    파이프라인 expand_masks가 .repeat(1,1,1,3)을 하므로 4D 필요
        mask_norm = (mask.astype(np.float32) / 255.0)  # (H, W)
        masks_np = np.stack([mask_norm] * num_frames)  # (T, H, W)
        masks_np = masks_np[..., np.newaxis]  # (T, H, W, 1)
        masks = torch.from_numpy(masks_np).to(dtype=torch.float32)

        # 3) 추론
        generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(
            f"Running MiniMax-Remover: {num_frames} frames, {w}x{h}, "
            f"steps={num_inference_steps}, seed={seed}, "
            f"images={images.shape}, masks={masks.shape}"
        )

        result = self.pipe(
            images=images,
            masks=masks,
            num_frames=num_frames,
            height=h,
            width=w,
            num_inference_steps=num_inference_steps,
            generator=generator,
            iterations=mask_dilation,
        ).frames[0]  # (T, H, W, 3) float32 [0,1] or list of PIL

        # 4) 출력 → BGR numpy [0,255]
        output_frames = []
        if isinstance(result, np.ndarray):
            # output_type="np": (T, H, W, 3) float32 [0,1]
            for frame in result:
                rgb = (frame * 255).clip(0, 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                output_frames.append(bgr)
        else:
            # output_type="pil": list of PIL Image
            for pil_img in result:
                rgb = np.array(pil_img, dtype=np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                output_frames.append(bgr)

        return output_frames


# 글로벌 모델 인스턴스
model: MiniMaxRemoverModel | None = None

# SAM2 predictor 싱글톤
_sam2_predictor = None
_sam2_image_id = None  # 이미지 임베딩 캐시

# RVM 모델 싱글톤
_rvm_model = None

# 동시 추론 방지용 락
_inference_lock = asyncio.Lock()
_sam2_lock = asyncio.Lock()
_rvm_lock = asyncio.Lock()


def _get_rvm_model(device: str = "cuda"):
    """RobustVideoMatting 모델을 lazy loading한다 (torch.hub 사용)."""
    global _rvm_model
    if _rvm_model is None:
        logger.info("Loading RVM model via torch.hub...")
        _rvm_model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            "mobilenetv3",
            trust_repo=True,
        ).eval().to(device)
        logger.info("RVM model loaded.")
    return _rvm_model


@app.on_event("startup")
async def load_models():
    """서버 시작 시 모델을 GPU에 로드한다."""
    global model
    weights_dir = os.environ.get(
        "WEIGHTS_DIR", "/workspace/weights/minimax-remover"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading MiniMax-Remover models...")
    model = MiniMaxRemoverModel(weights_dir=weights_dir, device=device)
    logger.info("Models loaded successfully.")

    # SAM2 로드
    global _sam2_predictor
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        logger.info("Loading SAM2 model...")
        _sam2_predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2.1-hiera-tiny", device=device,
        )
        logger.info("SAM2 model loaded.")
    except Exception as e:
        logger.warning(f"SAM2 load failed (segmentation unavailable): {e}")

    # 워밍업: 작은 더미 추론으로 CUDA 커널 컴파일
    logger.info("Warming up with dummy inference...")
    dummy_frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(4)]
    dummy_mask = np.zeros((64, 64), dtype=np.uint8)
    dummy_mask[20:40, 20:40] = 255
    try:
        model.inpaint(
            dummy_frames, dummy_mask,
            num_inference_steps=1, seed=0, mask_dilation=1,
        )
        logger.info("Warmup complete.")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")


@app.get("/health")
async def health_check():
    """서버 상태 확인."""
    gpu_available = torch.cuda.is_available()
    model_loaded = model is not None
    gpu_mem = {}
    if gpu_available:
        gpu_mem = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
            "total_mb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2, 1
            ),
            "gpu_name": torch.cuda.get_device_name(0),
        }
    return {
        "status": "healthy" if (gpu_available and model_loaded) else "degraded",
        "gpu_available": gpu_available,
        "model_loaded": model_loaded,
        "gpu_memory": gpu_mem,
        "engine": "minimax-remover",
        "timestamp": time.time(),
    }


@app.post("/inpaint")
async def inpaint_chunk(
    video: UploadFile = File(..., description="비디오 청크 (.mp4)"),
    mask: UploadFile = File(..., description="바이너리 마스크 (.png)"),
    max_inpaint_height: int = Form(480),
    feather_px: int = Form(5),
    mask_dilation: int = Form(6),
    num_inference_steps: int = Form(12),
    seed: int = Form(42),
):
    """
    비디오 청크를 인페인팅한다.

    해상도 전략:
    1. 원본 높이 > max_inpaint_height → 다운스케일 후 처리 → 업스케일 합성
    2. 원본 높이 <= max_inpaint_height → 원본 해상도로 직접 처리
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    async with _inference_lock:
        try:
            return await _do_inpaint(
                video, mask,
                max_inpaint_height, feather_px,
                mask_dilation, num_inference_steps, seed,
            )
        except torch.cuda.OutOfMemoryError as e:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            raise HTTPException(status_code=500, detail=f"GPU 메모리 부족: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()


async def _do_inpaint(
    video: UploadFile,
    mask: UploadFile,
    max_inpaint_height: int,
    feather_px: int,
    mask_dilation: int,
    num_inference_steps: int,
    seed: int,
) -> Response:
    """인페인팅 핵심 로직."""
    # 1) 비디오 디코딩
    video_bytes = await video.read()
    frames, fps = _decode_video_bytes(video_bytes)
    logger.info(
        f"Decoded {len(frames)} frames, "
        f"{frames[0].shape[1]}x{frames[0].shape[0]}, fps={fps:.1f}"
    )

    # 2) 마스크 디코딩
    mask_bytes = await mask.read()
    mask_array = _decode_mask_bytes(mask_bytes)

    # 크기 검증
    orig_h, orig_w = frames[0].shape[:2]
    mask_h, mask_w = mask_array.shape[:2]
    if (orig_h, orig_w) != (mask_h, mask_w):
        mask_array = cv2.resize(
            mask_array, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )

    # 3) 해상도 판단
    needs_upscale = orig_h > max_inpaint_height
    original_frames = None
    original_mask = None

    if needs_upscale:
        original_frames = [f.copy() for f in frames]
        original_mask = mask_array.copy()
        frames, mask_array, proc_h, proc_w = _downscale_for_processing(
            frames, mask_array, max_inpaint_height
        )
        logger.info(f"Downscaled to {proc_w}x{proc_h} for processing")

    # 4) MiniMax-Remover 추론
    logger.info("Starting MiniMax-Remover inference...")
    t0 = time.time()
    inpainted = model.inpaint(
        frames=frames,
        mask=mask_array,
        num_inference_steps=num_inference_steps,
        seed=seed,
        mask_dilation=mask_dilation,
    )
    logger.info(f"Inference done in {time.time() - t0:.1f}s")

    # 5) 업스케일 + 합성 (필요한 경우)
    if needs_upscale:
        logger.info("Upscaling and compositing...")
        inpainted = _upscale_and_composite(
            inpainted, original_frames, original_mask,
            orig_h, orig_w, feather_px,
        )

    # 6) mp4 인코딩
    result_bytes = _encode_frames_to_mp4(inpainted, fps)
    logger.info(f"Encoded result: {len(result_bytes)} bytes")

    return Response(
        content=result_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=inpainted.mp4"},
    )


@app.post("/rvm_matting")
async def rvm_matting(
    video: UploadFile = File(..., description="입력 비디오 (.mp4)"),
    background: UploadFile = File(..., description="클린 레퍼런스 배경 이미지 (.png)"),
    downsample_ratio: float = Form(0.25),
):
    """
    RVM으로 비디오에서 전경 알파 마스크를 추출한다.

    입력:
      - video: 원본 비디오 프레임들
      - background: 클린 레퍼런스 배경 (LaMa 인페인팅 결과)
      - downsample_ratio: RVM 처리 해상도 비율 (기본 0.25)

    반환: 알파 채널 그레이스케일 mp4 (프레임별 전경 확률 0~255)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    async with _rvm_lock:
        try:
            return await _do_rvm_matting(video, background, downsample_ratio, device)
        except Exception as e:
            logger.error(f"RVM matting error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


async def _do_rvm_matting(
    video: UploadFile,
    background: UploadFile,
    downsample_ratio: float,
    device: str,
) -> Response:
    """RVM 매팅 핵심 로직."""
    import torch
    import torchvision.transforms.functional as TF

    rvm = _get_rvm_model(device)

    # 1) 비디오 디코딩
    video_bytes = await video.read()
    frames_bgr, fps = _decode_video_bytes(video_bytes)
    h, w = frames_bgr[0].shape[:2]

    # 2) 배경 이미지 디코딩
    bg_bytes = await background.read()
    bg_arr = np.frombuffer(bg_bytes, dtype=np.uint8)
    bg_bgr = cv2.imdecode(bg_arr, cv2.IMREAD_COLOR)
    if bg_bgr is None:
        raise HTTPException(status_code=400, detail="배경 이미지를 읽을 수 없습니다")
    # 배경을 비디오 크기에 맞춤
    if bg_bgr.shape[:2] != (h, w):
        bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_AREA)
    bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)

    # 배경 텐서: (1, 3, H, W) float32 [0,1]
    bg_tensor = torch.from_numpy(bg_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # 3) RVM 추론: recurrent state 초기화
    rec = [None] * 4  # h0, h1, h2, h3

    alpha_frames = []
    with torch.no_grad():
        for frame_bgr in frames_bgr:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            src = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            # RVM forward: src, bgr, recurrent states, downsample_ratio
            fgr, pha, *rec = rvm(src, bg_tensor, *rec, downsample_ratio=downsample_ratio)
            # pha: (1, 1, H, W) float32 [0,1]
            alpha_np = (pha.squeeze().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            alpha_frames.append(alpha_np)

    # 4) 알파 그레이스케일 프레임 → BGR(3ch)로 변환해서 mp4 인코딩
    alpha_bgr = [cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) for a in alpha_frames]
    result_bytes = _encode_frames_to_mp4(alpha_bgr, fps)
    logger.info(f"RVM matting done: {len(alpha_frames)} frames, {len(result_bytes)} bytes")

    return Response(
        content=result_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=alpha.mp4"},
    )


@app.post("/segment")
async def segment_image(
    image: UploadFile = File(..., description="RGB 이미지 (.png/.jpg)"),
    positive_points: str = Form("[]", description="제거 대상 좌표 [[x,y],...]"),
    negative_points: str = Form("[]", description="보존 대상 좌표 [[x,y],...]"),
):
    """
    SAM2로 이미지에서 객체를 세그멘테이션한다.
    Returns: PNG 바이너리 마스크 (255=제거, 0=유지)
    """
    import json

    if _sam2_predictor is None:
        raise HTTPException(status_code=503, detail="SAM2 model not loaded")

    async with _sam2_lock:
        try:
            return await _do_segment(image, positive_points, negative_points)
        except Exception as e:
            logger.error(f"Segment error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


async def _do_segment(
    image: UploadFile,
    positive_points_str: str,
    negative_points_str: str,
) -> Response:
    """SAM2 세그멘테이션 핵심 로직."""
    import json

    global _sam2_image_id

    # 1) 이미지 디코딩
    img_bytes = await image.read()
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2) 포인트 파싱
    pos_pts = json.loads(positive_points_str)
    neg_pts = json.loads(negative_points_str)

    if not pos_pts:
        raise HTTPException(status_code=400, detail="positive_points가 비어있습니다")

    all_points = pos_pts + neg_pts
    labels = [1] * len(pos_pts) + [0] * len(neg_pts)

    point_coords = np.array(all_points, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)

    # 3) 이미지 임베딩 (캐시)
    image_id = (img_rgb.shape, img_rgb[0, 0, 0].item(), img_rgb[-1, -1, -1].item())

    with torch.inference_mode():
        if image_id != _sam2_image_id:
            logger.info(f"SAM2: computing image embeddings {img_rgb.shape}...")
            t0 = time.time()
            _sam2_predictor.set_image(img_rgb)
            _sam2_image_id = image_id
            logger.info(f"SAM2: embeddings done in {time.time() - t0:.2f}s")
        else:
            logger.info("SAM2: using cached embeddings")

        masks, scores, _ = _sam2_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    best_idx = np.argmax(scores)
    best_mask = (masks[best_idx].astype(np.uint8) * 255)

    logger.info(
        f"SAM2: {len(pos_pts)} pos, {len(neg_pts)} neg → "
        f"score={scores[best_idx]:.3f}, "
        f"coverage={masks[best_idx].sum() / masks[best_idx].size:.1%}"
    )

    # 4) PNG 인코딩
    _, png_bytes = cv2.imencode(".png", best_mask)

    return Response(
        content=png_bytes.tobytes(),
        media_type="image/png",
    )


# ── 유틸리티 함수 ──


def _decode_video_bytes(video_bytes: bytes) -> tuple:
    """mp4 바이트 → (프레임 리스트, fps)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    finally:
        os.unlink(tmp_path)

    if not frames:
        raise HTTPException(
            status_code=400, detail="비디오에서 프레임을 읽을 수 없습니다"
        )

    return frames, fps


def _decode_mask_bytes(mask_bytes: bytes) -> np.ndarray:
    """PNG 바이트 → HxW uint8 바이너리 마스크."""
    arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(
            status_code=400, detail="마스크 이미지를 읽을 수 없습니다"
        )
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary


def _downscale_for_processing(
    frames: list,
    mask: np.ndarray,
    max_height: int,
) -> tuple:
    """프레임과 마스크를 다운스케일한다."""
    orig_h, orig_w = frames[0].shape[:2]
    scale = max_height / orig_h
    new_w = round(orig_w * scale)
    new_h = max_height

    # 8의 배수로 맞춤
    new_h = (new_h // 8) * 8
    new_w = (new_w // 8) * 8

    resized_frames = [
        cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)
        for f in frames
    ]
    resized_mask = cv2.resize(
        mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )

    return resized_frames, resized_mask, new_h, new_w


def _upscale_and_composite(
    inpainted: list,
    originals: list,
    mask: np.ndarray,
    orig_h: int,
    orig_w: int,
    feather_radius: int = 5,
) -> list:
    """
    인페인팅 결과를 원본 해상도로 업스케일하고 Hard Mask Composite을 적용한다.

    마스크 바깥은 bit-for-bit 원본 보존.
    경계는 feather 블렌딩으로 자연스러운 전환.
    """
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if feather_radius > 0:
        ksize = feather_radius * 2 + 1
        feather_mask = cv2.GaussianBlur(
            binary.astype(np.float32) / 255.0,
            (ksize, ksize),
            sigmaX=feather_radius / 2,
        )
    else:
        feather_mask = binary.astype(np.float32) / 255.0

    feather_3ch = feather_mask[:, :, np.newaxis]  # (H, W, 1)

    result = []
    for inp_frame, orig_frame in zip(inpainted, originals):
        upscaled = cv2.resize(
            inp_frame, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
        )
        composite = (
            upscaled.astype(np.float32) * feather_3ch
            + orig_frame.astype(np.float32) * (1.0 - feather_3ch)
        )
        result.append(np.clip(composite, 0, 255).astype(np.uint8))

    return result


def _encode_frames_to_mp4(frames: list, fps: float = 30.0) -> bytes:
    """BGR numpy 프레임 리스트 → mp4 바이트."""
    buf = io.BytesIO()
    rgb_frames = [f[:, :, ::-1] for f in frames]
    imageio.mimwrite(
        buf,
        rgb_frames,
        format="mp4",
        fps=fps,
        codec="libx264",
        quality=8,
    )
    return buf.getvalue()
