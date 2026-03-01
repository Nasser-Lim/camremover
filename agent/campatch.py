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
    RVM으로 알파 마스크를 추출하고, 클린 레퍼런스 + 원본 프레임을 합성한다.

    합성 공식:
      result = clean_ref × (1 - pha_combined) + original × pha_combined

    여기서 pha_combined = max(feathered_mask 내 강제 블렌딩, RVM_pha 보호 영역)

    즉:
      - 마스크 영역: clean_ref로 완전 교체 (pha_combined = feathered_mask)
      - 마스크 바깥 피사체: original 유지 (RVM pha가 높은 곳)
    """
    import cv2 as _cv2

    # 1) 배경 이미지를 임시 파일로 저장
    bg_path = os.path.join(temp_dir, "clean_ref_bg.png")
    bg_rgb = _cv2.cvtColor(clean_ref_bgr, _cv2.COLOR_BGR2RGB)
    from PIL import Image as _Img
    _Img.fromarray(bg_rgb).save(bg_path)

    # 2) Pod에 RVM 요청
    client = RunPodClient(config.runpod)
    _report("blending", "RVM 알파 추출 중 (Pod 연결)...", percent=26)
    alpha_mp4_bytes = client.rvm_matting(
        video_path=video_path,
        background_path=bg_path,
        downsample_ratio=config.campatch.rvm_downsample_ratio,
    )
    client.close()

    # 3) 알파 mp4 디코딩
    alpha_tmp = os.path.join(temp_dir, "alpha.mp4")
    with open(alpha_tmp, "wb") as f:
        f.write(alpha_mp4_bytes)

    alpha_cap = _cv2.VideoCapture(alpha_tmp)
    alpha_frames_raw = []
    while True:
        ret, frame = alpha_cap.read()
        if not ret:
            break
        # BGR→Grayscale (3ch로 인코딩된 알파를 단채널로)
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
        alpha_frames_raw.append(gray.astype(np.float32) / 255.0)
    alpha_cap.release()

    logger.info(f"RVM 알파 프레임: {len(alpha_frames_raw)}")

    # 4) 원본 프레임 읽기 + 합성
    cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
    blended_frames = []
    frame_idx = 0

    hard_mask = (feathered_mask > 0.01).astype(np.float32)  # 마스크 내부

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx < len(alpha_frames_raw):
            rvm_pha = alpha_frames_raw[frame_idx]  # 전경 확률 [0,1]
        else:
            rvm_pha = np.zeros(frame_bgr.shape[:2], dtype=np.float32)

        # 마스크 내부: clean_ref 강제 사용 (feathered_mask)
        # 마스크 외부 피사체: original 유지 (rvm_pha * (1 - hard_mask))
        # pha_combined = 마스크 내부 → feathered_mask 값 (clean_ref 교체)
        #                마스크 외부 → rvm_pha (피사체 있으면 original 유지)
        pha_outside = rvm_pha * (1.0 - hard_mask)  # 마스크 밖 RVM 피사체
        # 최종 alpha: 마스크 안은 feathered_mask(clean_ref 비율), 밖은 0 + pha_outside(original 비율)
        # → result = clean_ref × (1 - alpha) + original × alpha
        alpha = feathered_mask * hard_mask + pha_outside
        alpha = np.clip(alpha, 0.0, 1.0)

        alpha_3d = alpha[:, :, np.newaxis]
        blended = (
            clean_ref_bgr.astype(np.float32) * (1.0 - alpha_3d)
            + frame_bgr.astype(np.float32) * alpha_3d
        )
        blended_frames.append(np.clip(blended, 0, 255).astype(np.uint8))

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
