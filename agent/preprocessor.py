"""마스크 전처리, 영상 메타데이터 추출, 청크 분할 모듈."""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .config import MaskConfig, VideoConfig

logger = logging.getLogger("camremover.preprocessor")


@dataclass
class VideoChunk:
    """영상 청크 정보."""

    chunk_index: int
    frame_start: int  # 시작 프레임 (포함)
    frame_end: int  # 끝 프레임 (미포함)
    overlap_start: int  # 앞쪽 겹침 프레임 수
    overlap_end: int  # 뒤쪽 겹침 프레임 수
    file_path: str  # 임시 mp4 파일 경로
    total_frames: int  # 이 청크의 프레임 수
    padded_frames: int = 0  # 마지막 프레임 복제로 추가된 패딩 프레임 수


@dataclass
class VideoInfo:
    """영상 메타데이터."""

    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    has_audio: bool
    file_path: str


def extract_first_frame(video_path: str) -> np.ndarray:
    """
    영상의 첫 프레임을 추출한다.

    Returns:
        HxWx3 uint8 BGR numpy 배열
    """
    return extract_frame_at(video_path, 0)


def extract_frame_at(video_path: str, frame_idx: int) -> np.ndarray:
    """
    영상의 특정 프레임을 추출한다.

    Returns:
        HxWx3 uint8 BGR numpy 배열
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"프레임 {frame_idx}을 읽을 수 없습니다: {video_path}")

    return frame


def get_video_info(video_path: str) -> VideoInfo:
    """영상 메타데이터를 추출한다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = total_frames / fps if fps > 0 else 0

    # ffprobe로 오디오 스트림 존재 여부 확인
    has_audio = _check_audio_stream(video_path)

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration_seconds=duration,
        has_audio=has_audio,
        file_path=video_path,
    )


def _check_audio_stream(video_path: str) -> bool:
    """ffprobe로 오디오 스트림 존재 여부를 확인한다."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "json",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            return len(streams) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return False


def preprocess_mask(
    raw_mask: np.ndarray,
    target_size: Tuple[int, int],
    config: MaskConfig,
) -> np.ndarray:
    """
    Gradio ImageEditor 출력에서 바이너리 마스크를 생성한다.

    Args:
        raw_mask: Gradio ImageEditor의 layers[0] (HxWx4 RGBA 또는 HxWx3 RGB 또는 HxW)
        target_size: (width, height) 영상 크기
        config: 마스크 전처리 설정

    Returns:
        HxW uint8 바이너리 마스크 (255=제거, 0=유지)
    """
    # 1) 마스크 추출
    if raw_mask.ndim == 3 and raw_mask.shape[2] == 4:
        # RGBA: alpha 채널 사용
        mask = raw_mask[:, :, 3]
    elif raw_mask.ndim == 3 and raw_mask.shape[2] == 3:
        # RGB: 아무 채널이든 0이 아니면 마스크
        mask = np.any(raw_mask > 0, axis=2).astype(np.uint8) * 255
    elif raw_mask.ndim == 2:
        mask = raw_mask
    else:
        raise ValueError(f"지원하지 않는 마스크 형식: shape={raw_mask.shape}")

    # 2) 바이너리 변환
    _, mask = cv2.threshold(mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    # 3) 영상 크기로 리사이즈
    target_w, target_h = target_size
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(
            mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )

    # 4) 마스크 팽창 (dilation)
    if config.dilation_iterations > 0 and config.dilation_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.dilation_kernel_size, config.dilation_kernel_size),
        )
        mask = cv2.dilate(mask, kernel, iterations=config.dilation_iterations)

    return mask


def chunk_video(
    video_path: str,
    chunk_size: int,
    overlap: int,
    output_dir: str,
) -> List[VideoChunk]:
    """
    영상을 겹치는 청크로 분할한다.

    stride = chunk_size - overlap
    예: 250프레임, chunk=80, overlap=10
      Chunk0: [0, 80)   overlap_start=0,  overlap_end=10
      Chunk1: [70, 150) overlap_start=10, overlap_end=10
      Chunk2: [140, 220) overlap_start=10, overlap_end=10
      Chunk3: [210, 250) overlap_start=10, overlap_end=0
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stride = chunk_size - overlap

    # 청크 경계 계산
    chunk_boundaries = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        chunk_boundaries.append((start, end))
        if end >= total_frames:
            break
        start += stride

    # 프레임을 모두 읽은 뒤 청크별로 저장
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    actual_total = len(all_frames)
    logger.info(f"Read {actual_total} frames, splitting into {len(chunk_boundaries)} chunks")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunks = []
    for idx, (start, end) in enumerate(chunk_boundaries):
        end = min(end, actual_total)
        chunk_frames = all_frames[start:end]

        # 겹침 정보
        overlap_start = overlap if idx > 0 else 0
        overlap_end = overlap if idx < len(chunk_boundaries) - 1 else 0

        # 마지막 청크가 겹침보다 짧으면 겹침 조정
        if len(chunk_frames) <= overlap_start:
            continue

        # 프레임 수가 chunk_size보다 적으면 마지막 프레임을 복제하여 패딩
        padded = 0
        if len(chunk_frames) < chunk_size:
            padded = chunk_size - len(chunk_frames)
            last_frame = chunk_frames[-1]
            chunk_frames = chunk_frames + [last_frame] * padded

        # mp4로 저장
        chunk_path = str(output_path / f"chunk_{idx:04d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(chunk_path, fourcc, fps, (width, height))
        for f in chunk_frames:
            writer.write(f)
        writer.release()

        chunks.append(
            VideoChunk(
                chunk_index=idx,
                frame_start=start,
                frame_end=end,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
                file_path=chunk_path,
                total_frames=len(chunk_frames),
                padded_frames=padded,
            )
        )

    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def save_mask_as_png(mask: np.ndarray, output_path: str) -> str:
    """바이너리 마스크를 PNG로 저장한다."""
    cv2.imwrite(output_path, mask)
    return output_path


