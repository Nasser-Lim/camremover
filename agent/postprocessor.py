"""청크 합치기, 오디오 복원, 최종 출력 생성 모듈."""

import logging
import subprocess
from pathlib import Path
from typing import List

import numpy as np

from .config import VideoConfig
from .preprocessor import VideoChunk, VideoInfo

logger = logging.getLogger("camremover.postprocessor")


def merge_chunks(
    chunk_results: List[dict],
    video_info: VideoInfo,
    output_path: str,
    config: VideoConfig,
) -> str:
    """
    인페인팅된 청크들을 크로스페이드 블렌딩으로 합친다.

    chunk_results: [{"chunk": VideoChunk, "result_path": str}, ...]
    """
    if not chunk_results:
        raise ValueError("합칠 청크가 없습니다")

    # 정렬
    chunk_results.sort(key=lambda x: x["chunk"].chunk_index)

    all_frames: List[np.ndarray] = []

    for i, cr in enumerate(chunk_results):
        chunk: VideoChunk = cr["chunk"]
        frames = _decode_video(cr["result_path"])

        # 패딩된 프레임 제거
        if chunk.padded_frames > 0 and len(frames) > chunk.padded_frames:
            frames = frames[: len(frames) - chunk.padded_frames]

        if not frames:
            logger.warning(f"Chunk {chunk.chunk_index}: 프레임이 없습니다")
            continue

        if i == 0:
            # 첫 청크: 전체 사용
            all_frames.extend(frames)
        else:
            overlap = chunk.overlap_start
            if overlap > 0 and overlap <= len(frames) and overlap <= len(all_frames):
                # 겹치는 구간: 크로스페이드
                blended = _crossfade_frames(
                    all_frames[-overlap:],
                    frames[:overlap],
                    overlap,
                )
                # 기존 끝부분을 블렌딩된 프레임으로 교체
                all_frames[-overlap:] = blended
                # 비겹침 구간 추가
                all_frames.extend(frames[overlap:])
            else:
                # 겹침 없이 이어붙이기
                all_frames.extend(frames)

    logger.info(f"Merged {len(all_frames)} frames from {len(chunk_results)} chunks")

    # FFmpeg pipe로 인코딩
    _encode_video_ffmpeg(all_frames, video_info.fps, output_path, config)
    return output_path


def _crossfade_frames(
    frames_a: List[np.ndarray],
    frames_b: List[np.ndarray],
    num_overlap: int,
) -> List[np.ndarray]:
    """
    두 프레임 시퀀스를 크로스페이드 블렌딩한다.

    frames_a: 이전 청크의 마지막 num_overlap 프레임
    frames_b: 현재 청크의 처음 num_overlap 프레임
    """
    blended = []
    for i in range(num_overlap):
        alpha = (i + 0.5) / num_overlap
        result = (
            frames_a[i].astype(np.float32) * (1 - alpha)
            + frames_b[i].astype(np.float32) * alpha
        )
        blended.append(np.clip(result, 0, 255).astype(np.uint8))
    return blended


def restore_audio(
    original_video: str,
    inpainted_video: str,
    output_path: str,
    config: VideoConfig,
) -> str:
    """원본 영상에서 오디오를 추출하여 인페인팅된 영상에 합성한다."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", inpainted_video,
        "-i", original_video,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", config.output_codec,
        "-crf", str(config.output_crf),
        "-preset", config.output_preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error(f"FFmpeg error: {result.stderr}")
        raise RuntimeError(f"FFmpeg 오디오 합성 실패: {result.stderr[-500:]}")

    return output_path


def _decode_video(video_path: str) -> List[np.ndarray]:
    """mp4 파일에서 프레임을 읽는다."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _encode_video_ffmpeg(
    frames: List[np.ndarray],
    fps: float,
    output_path: str,
    config: VideoConfig,
) -> None:
    """FFmpeg subprocess pipe로 프레임을 mp4로 인코딩한다."""
    if not frames:
        raise ValueError("인코딩할 프레임이 없습니다")

    h, w = frames[0].shape[:2]

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", config.output_codec,
        "-crf", str(config.output_crf),
        "-preset", config.output_preset,
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass

    stderr_bytes = proc.stderr.read()
    proc.wait(timeout=300)

    if proc.returncode != 0:
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        logger.error(f"FFmpeg encode error: {stderr}")
        raise RuntimeError(f"FFmpeg 인코딩 실패: {stderr[-500:]}")
