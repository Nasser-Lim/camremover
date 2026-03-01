"""CamPatch 엔진 — LaMa 이미지 인페인팅 + 패치 블렌딩."""

import logging
import os
import tempfile
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from .config import AppConfig
from .main import ProcessingProgress
from .postprocessor import restore_audio, _encode_video_ffmpeg
from .preprocessor import get_video_info, preprocess_mask
from .runpod_client import RunPodClient

logger = logging.getLogger("camremover.campatch")

_lama_model = None


def _get_lama():
    """LaMa 모델을 lazy loading한다. 첫 호출 시 모델 다운로드."""
    global _lama_model
    if _lama_model is None:
        from simple_lama_inpainting import SimpleLama

        logger.info("LaMa 모델 로딩 중...")
        _lama_model = SimpleLama()
        logger.info("LaMa 모델 로딩 완료")
    return _lama_model


def inpaint_single_frame(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    LaMa로 단일 프레임을 인페인팅한다.

    Args:
        frame_rgb: HxWx3 uint8 RGB
        mask: HxW uint8 바이너리 마스크 (255=인페인팅, 0=유지)

    Returns:
        HxWx3 uint8 RGB 인페인팅 결과
    """
    lama = _get_lama()
    result = lama(Image.fromarray(frame_rgb), Image.fromarray(mask))
    return np.array(result)


def create_feathered_mask(mask: np.ndarray, feather_radius: int) -> np.ndarray:
    """
    바이너리 마스크에 가우시안 페더링을 적용한다.

    Returns:
        HxW float32 (0.0~1.0)
    """
    if feather_radius <= 0:
        return (mask / 255.0).astype(np.float32)

    ksize = feather_radius * 2 + 1
    blurred = cv2.GaussianBlur(
        mask.astype(np.float32), (ksize, ksize), sigmaX=feather_radius / 2
    )
    return blurred / 255.0


def blend_patch(
    original_bgr: np.ndarray,
    clean_ref_bgr: np.ndarray,
    feathered_mask: np.ndarray,
) -> np.ndarray:
    """
    원본 프레임에 클린 레퍼런스 패치를 페더링 블렌딩한다.

    Args:
        original_bgr: HxWx3 uint8 BGR
        clean_ref_bgr: HxWx3 uint8 BGR
        feathered_mask: HxW float32 (0.0~1.0)

    Returns:
        HxWx3 uint8 BGR
    """
    alpha_3d = feathered_mask[:, :, np.newaxis]
    blended = (
        original_bgr.astype(np.float32) * (1 - alpha_3d)
        + clean_ref_bgr.astype(np.float32) * alpha_3d
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _blend_with_rvm(
    video_path: str,
    clean_ref_bgr: np.ndarray,
    feathered_mask: np.ndarray,
    config: AppConfig,
    total_frames: int,
    cap: "cv2.VideoCapture",
    _report,
    temp_dir: str,
) -> list:
    """
    RVM으로 마스크 영역 내 피사체 알파를 추출하고 합성한다.

    전략:
      - 마스크 bbox 크롭 영상만 Pod에 전송 (전체 화면 불필요)
      - RVM 알파를 마스크 영역에만 적용
      - 마스크 밖은 원본 그대로 유지

    합성 공식 (마스크 bbox 내부):
      result = original × rvm_pha + clean_ref × (1 - rvm_pha)
      단, 마스크 완전 내부는 feathered_mask 비율로 clean_ref 보장
    """
    import cv2 as _cv2

    # 1) 마스크 bbox 계산 (RVM 크롭 범위)
    hard_mask = (feathered_mask > 0.01).astype(np.uint8)
    ys, xs = np.where(hard_mask)
    if len(ys) == 0:
        # 마스크 없음 — 단순 blend_patch와 동일
        cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
        blended_frames = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            blended_frames.append(blend_patch(frame_bgr, clean_ref_bgr, feathered_mask))
        cap.release()
        return blended_frames

    pad = 32  # 경계 여유
    h_full, w_full = feathered_mask.shape
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(h_full, int(ys.max()) + pad + 1)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(w_full, int(xs.max()) + pad + 1)
    logger.info(f"RVM 크롭 bbox: ({x1},{y1})-({x2},{y2}) / 전체: {w_full}x{h_full}")

    # 2) 크롭 영상 생성 (앞에 워밍업 프레임 추가)
    # RVM은 recurrent 모델이라 초반 프레임에서 배경 학습이 불안정 → 플리커링 발생
    # 첫 프레임을 N번 반복해서 앞에 붙이면 RVM이 배경을 미리 수렴시킴
    RVM_WARMUP_FRAMES = 15
    crop_video_path = os.path.join(temp_dir, "crop_input.mp4")
    cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(_cv2.CAP_PROP_FPS) or 30.0
    cw, ch = x2 - x1, y2 - y1

    # 첫 프레임 읽기 (워밍업용)
    ret, first_frame = cap.read()
    if not ret:
        first_frame = None
    first_crop = first_frame[y1:y2, x1:x2] if first_frame is not None else None
    cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)

    writer = _cv2.VideoWriter(
        crop_video_path,
        _cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (cw, ch),
    )
    # 워밍업 프레임 삽입
    if first_crop is not None:
        for _ in range(RVM_WARMUP_FRAMES):
            writer.write(first_crop)
    # 실제 프레임
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame[y1:y2, x1:x2])
    writer.release()
    logger.info(f"RVM 크롭 영상 생성: {RVM_WARMUP_FRAMES}개 워밍업 프레임 + 실제 프레임")

    # 3) 크롭 배경 이미지 저장
    bg_path = os.path.join(temp_dir, "clean_ref_crop.png")
    crop_ref = clean_ref_bgr[y1:y2, x1:x2]
    from PIL import Image as _Img
    _Img.fromarray(_cv2.cvtColor(crop_ref, _cv2.COLOR_BGR2RGB)).save(bg_path)

    # 4) Pod에 크롭 영상으로 RVM 요청
    client = RunPodClient(config.runpod)
    _report("blending", "RVM 알파 추출 중 (마스크 영역)...", percent=26)
    alpha_mp4_bytes = client.rvm_matting(
        video_path=crop_video_path,
        background_path=bg_path,
        downsample_ratio=config.campatch.rvm_downsample_ratio,
    )
    client.close()

    # 5) 알파 mp4 디코딩
    alpha_tmp = os.path.join(temp_dir, "alpha_crop.mp4")
    with open(alpha_tmp, "wb") as f:
        f.write(alpha_mp4_bytes)

    alpha_cap = _cv2.VideoCapture(alpha_tmp)
    alpha_frames_raw = []
    while True:
        ret, frame = alpha_cap.read()
        if not ret:
            break
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
        alpha_frames_raw.append(gray.astype(np.float32) / 255.0)
    alpha_cap.release()
    logger.info(f"RVM 알파 프레임 (워밍업 포함): {len(alpha_frames_raw)}")

    # 워밍업 프레임 제거 (앞에 붙인 N개)
    alpha_frames_raw = alpha_frames_raw[RVM_WARMUP_FRAMES:]
    logger.info(f"RVM 알파 프레임 (워밍업 제거 후): {len(alpha_frames_raw)}")

    # 6) 크롭 영역의 feathered_mask / hard_mask
    feathered_crop = feathered_mask[y1:y2, x1:x2]
    hard_crop = hard_mask[y1:y2, x1:x2].astype(np.float32)
    clean_ref_crop = clean_ref_bgr[y1:y2, x1:x2]

    # 7) 전체 프레임 읽기 + 합성 (마스크 bbox 영역만 교체)
    cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
    blended_frames = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        result = frame_bgr.copy()

        # 크롭 영역만 RVM 알파 적용
        if frame_idx < len(alpha_frames_raw):
            rvm_pha = alpha_frames_raw[frame_idx]
            if rvm_pha.shape != (ch, cw):
                rvm_pha = _cv2.resize(rvm_pha, (cw, ch), interpolation=_cv2.INTER_LINEAR)
        else:
            rvm_pha = np.zeros((ch, cw), dtype=np.float32)

        # keep_original = 1 → 원본 유지, 0 → clean_ref 사용
        # 마스크 안: clean_ref 사용 (keep=0), 단 RVM 피사체는 원본 유지 (keep=pha)
        # 마스크 밖: 원본 그대로 (keep=1)
        keep_inside = rvm_pha * hard_crop  # 마스크 안에서 피사체만 원본
        keep_outside = 1.0 - hard_crop     # 마스크 밖은 전부 원본
        keep_original = np.clip(keep_inside + keep_outside, 0.0, 1.0)

        # 페더링 경계: feathered_mask의 그라디언트를 반영
        # feathered_crop이 0~1 사이인 경계에서 부드럽게 전환
        keep_original = np.maximum(keep_original, 1.0 - feathered_crop)

        k3 = keep_original[:, :, np.newaxis]
        orig_crop = frame_bgr[y1:y2, x1:x2].astype(np.float32)
        blended_crop = (
            orig_crop * k3
            + clean_ref_crop.astype(np.float32) * (1.0 - k3)
        )
        result[y1:y2, x1:x2] = np.clip(blended_crop, 0, 255).astype(np.uint8)

        blended_frames.append(result)

        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = 25 + (frame_idx / total_frames) * 50
            _report("blending", f"RVM 합성 중 ({frame_idx}/{total_frames})...", percent=pct)

    cap.release()
    return blended_frames


def generate_clean_reference(
    video_path: str,
    mask_raw: np.ndarray,
    config: AppConfig,
    ref_frame_idx: int = 0,
) -> np.ndarray:
    """
    클린 레퍼런스 이미지를 생성하여 RGB numpy array로 반환한다.
    UI 미리보기용.
    """
    video_info = get_video_info(video_path)
    mask = preprocess_mask(
        mask_raw,
        target_size=(video_info.width, video_info.height),
        config=config.mask,
    )
    mask_bool = mask > 0
    total_frames = video_info.total_frames
    ref_idx = max(0, min(int(ref_frame_idx), total_frames - 1))

    cap = cv2.VideoCapture(video_path)
    if ref_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ret, ref_frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"프레임 {ref_idx}을 읽을 수 없습니다")

    ref_rgb = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2RGB)
    inpaint_radius = max(config.campatch.feather_radius, 5)
    ksize = inpaint_radius * 2 + 1
    lama_mask = cv2.dilate(mask, np.ones((ksize, ksize), np.uint8))
    lama_result_rgb = inpaint_single_frame(ref_rgb, lama_mask)
    extra_bool = (lama_mask > 0) & ~mask_bool
    clean_ref_rgb = lama_result_rgb.copy()
    clean_ref_rgb[extra_bool] = ref_rgb[extra_bool]
    return clean_ref_rgb


def process_video_campatch(
    video_path: str,
    mask_raw: np.ndarray,
    config: AppConfig,
    ref_frame_idx: int = 0,
    progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
) -> str:
    """CamPatch 파이프라인을 실행한다."""
    temp_dir = tempfile.mkdtemp(prefix="campatch_")

    def _report(stage, message, **kwargs):
        if progress_callback:
            progress_callback(ProcessingProgress(stage=stage, message=message, **kwargs))

    try:
        # Stage 1: 영상 분석
        _report("preparing", "영상 분석 중...", percent=0)
        video_info = get_video_info(video_path)
        logger.info(
            f"Video: {video_info.width}x{video_info.height}, "
            f"{video_info.fps}fps, {video_info.total_frames} frames"
        )

        # Stage 2: 마스크 전처리
        _report("preparing", "마스크 전처리 중...", percent=5)
        mask = preprocess_mask(
            mask_raw,
            target_size=(video_info.width, video_info.height),
            config=config.mask,
        )
        coverage = mask.sum() / (mask.size * 255) * 100
        logger.info(f"Mask coverage: {coverage:.1f}%")

        # Stage 3: 기준 프레임으로 LaMa 클린 레퍼런스 생성
        mask_bool = mask > 0
        total_frames = video_info.total_frames
        ref_idx = max(0, min(int(ref_frame_idx), total_frames - 1))
        logger.info(f"기준 프레임: {ref_idx}")

        cap = cv2.VideoCapture(video_path)
        ret, ref_frame_bgr = cap.read()
        if not ret:
            cap.release()
            raise ValueError("첫 프레임을 읽을 수 없습니다")

        if ref_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
            ret, ref_frame_bgr = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, ref_frame_bgr = cap.read()

        _report("inpainting", "LaMa 인페인팅 중...", percent=15)
        ref_rgb = cv2.cvtColor(ref_frame_bgr, cv2.COLOR_BGR2RGB)
        inpaint_radius = max(config.campatch.feather_radius, 5)
        ksize = inpaint_radius * 2 + 1
        lama_mask = cv2.dilate(mask, np.ones((ksize, ksize), np.uint8))
        lama_result_rgb = inpaint_single_frame(ref_rgb, lama_mask)
        extra_bool = (lama_mask > 0) & ~mask_bool
        clean_ref_rgb = lama_result_rgb.copy()
        clean_ref_rgb[extra_bool] = ref_rgb[extra_bool]
        logger.info(f"클린 레퍼런스 생성 완료 (LaMa, inpaint_radius={inpaint_radius})")
        _report("inpainting", "클린 배경 생성 완료", percent=22)

        feathered_mask = create_feathered_mask(mask, config.campatch.feather_radius)
        clean_ref_bgr = cv2.cvtColor(clean_ref_rgb, cv2.COLOR_RGB2BGR)

        # Stage 4: 전 프레임 패치 블렌딩
        if config.campatch.rvm_enabled:
            _report("blending", "RVM 알파 마스크 추출 중...", percent=25)
            blended_frames = _blend_with_rvm(
                video_path, clean_ref_bgr, feathered_mask,
                config, total_frames, cap, _report, temp_dir,
            )
        else:
            _report("blending", "프레임 블렌딩 중...", percent=25)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            blended_frames = []

            frame_idx = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                blended_frames.append(blend_patch(frame_bgr, clean_ref_bgr, feathered_mask))
                frame_idx += 1
                if frame_idx % 30 == 0:
                    pct = 25 + (frame_idx / total_frames) * 50
                    _report("blending", f"프레임 블렌딩 중 ({frame_idx}/{total_frames})...", percent=pct)

            cap.release()

        logger.info(f"블렌딩 완료: {len(blended_frames)} frames")

        # Stage 5: FFmpeg 인코딩
        _report("encoding", "영상 인코딩 중...", percent=80)
        merged_path = os.path.join(temp_dir, "merged.mp4")
        _encode_video_ffmpeg(blended_frames, video_info.fps, merged_path, config.video)

        # Stage 6: 오디오 복원
        if video_info.has_audio:
            _report("audio", "오디오 복원 중...", percent=90)
            output_path = os.path.join(temp_dir, "output_final.mp4")
            restore_audio(video_path, merged_path, output_path, config.video)
        else:
            output_path = merged_path

        _report("done", "CamPatch 처리 완료!", percent=100)
        logger.info(f"Output: {output_path}")
        return output_path

    except Exception:
        logger.exception("CamPatch 처리 실패")
        raise
