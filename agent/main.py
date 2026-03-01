"""메인 오케스트레이터 — 전체 파이프라인을 조율한다."""

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import requests

from .config import AppConfig, load_config
from .postprocessor import merge_chunks, restore_audio
from .preprocessor import (
    VideoInfo,
    chunk_video,
    extract_first_frame,
    get_video_info,
    preprocess_mask,
    save_mask_as_png,
)
from .runpod_client import RunPodClient

logger = logging.getLogger("camremover.main")


@dataclass
class ProcessingProgress:
    """UI에 전달하는 진행 상태."""

    stage: str  # preparing, chunking, connecting, inpainting, merging, audio, done, error
    message: str
    chunk_current: int = 0
    chunk_total: int = 0
    percent: float = 0.0


class CamRemoverAgent:
    """거치카메라 제거 파이프라인 오케스트레이터."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.client = RunPodClient(self.config.runpod)
        self._temp_dir: Optional[str] = None

    def process_video(
        self,
        video_path: str,
        mask_raw: np.ndarray,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
    ) -> str:
        """
        전체 파이프라인을 실행한다.

        Args:
            video_path: 업로드된 영상 경로
            mask_raw: Gradio ImageEditor에서 받은 마스크
            progress_callback: 진행률 콜백

        Returns:
            최종 출력 영상 경로
        """
        try:
            self._temp_dir = tempfile.mkdtemp(prefix="camremover_")

            # ── Stage 1: 영상 분석 ──
            self._report(progress_callback, "preparing", "영상 분석 중...", percent=0)
            video_info = get_video_info(video_path)
            logger.info(
                f"Video: {video_info.width}x{video_info.height}, "
                f"{video_info.fps:.1f}fps, {video_info.total_frames} frames, "
                f"audio={video_info.has_audio}"
            )

            # ── Stage 2: 마스크 전처리 ──
            self._report(progress_callback, "preparing", "마스크 전처리 중...", percent=5)
            mask = preprocess_mask(
                mask_raw,
                target_size=(video_info.width, video_info.height),
                config=self.config.mask,
            )
            mask_path = os.path.join(self._temp_dir, "mask.png")
            save_mask_as_png(mask, mask_path)

            mask_coverage = (mask > 127).sum() / (mask.shape[0] * mask.shape[1])
            logger.info(f"Mask coverage: {mask_coverage:.1%}")

            # ── Stage 3: 영상 청크 분할 ──
            self._report(progress_callback, "chunking", "영상을 청크로 분할 중...", percent=10)
            chunks = chunk_video(
                video_path,
                chunk_size=self.config.video.chunk_size,
                overlap=self.config.video.chunk_overlap,
                output_dir=os.path.join(self._temp_dir, "chunks"),
            )
            total_chunks = len(chunks)
            logger.info(f"Split into {total_chunks} chunks")

            # ── Stage 4: Pod 연결 확인 ──
            self._report(progress_callback, "connecting", "GPU Pod에 연결 중...", percent=12)
            ready = self.client.wait_for_ready(
                max_wait_seconds=300,
                progress_callback=lambda msg: self._report(
                    progress_callback, "connecting", msg, percent=13,
                ),
            )
            if not ready:
                raise ConnectionError("GPU Pod에 연결할 수 없습니다 (타임아웃)")

            # RVM이 로드되어 있을 수 있으므로 MiniMax 추론 전 GPU 메모리 확보
            self.client.unload_model("rvm")
            self.client.reload_model("minimax")

            # ── Stage 5: 청크별 인페인팅 ──
            chunk_results = []
            for i, chunk in enumerate(chunks):
                pct = 15 + (i / total_chunks) * 65  # 15~80%
                self._report(
                    progress_callback,
                    "inpainting",
                    f"인페인팅 중 ({i + 1}/{total_chunks})...",
                    chunk_current=i + 1,
                    chunk_total=total_chunks,
                    percent=pct,
                )

                result_bytes = self._inpaint_with_retry(chunk.file_path, mask_path)

                result_path = os.path.join(
                    self._temp_dir, f"result_{i:04d}.mp4"
                )
                with open(result_path, "wb") as f:
                    f.write(result_bytes)

                chunk_results.append(
                    {"chunk": chunk, "result_path": result_path}
                )
                logger.info(
                    f"Chunk {i + 1}/{total_chunks} done "
                    f"({len(result_bytes)} bytes)"
                )

            # ── Stage 6: 청크 합치기 ──
            self._report(progress_callback, "merging", "청크 합치는 중...", percent=82)
            merged_path = os.path.join(self._temp_dir, "merged.mp4")
            merge_chunks(chunk_results, video_info, merged_path, self.config.video)

            # ── Stage 7: 오디오 복원 ──
            self._report(progress_callback, "audio", "오디오 복원 중...", percent=92)
            output_path = os.path.join(self._temp_dir, "output_final.mp4")
            restore_audio(
                video_path, merged_path, output_path, self.config.video
            )

            # ── 완료 ──
            self._report(progress_callback, "done", "처리 완료!", percent=100)
            logger.info(f"Output: {output_path}")
            return output_path

        except Exception as e:
            logger.exception("처리 실패")
            self._report(progress_callback, "error", str(e))
            raise

    def _inpaint_with_retry(
        self,
        chunk_path: str,
        mask_path: str,
    ) -> bytes:
        """인페인팅 요청 (재시도 없음 — 실패 시 즉시 에러)."""
        return self.client.inpaint_chunk(
            chunk_path,
            mask_path,
            self.config.minimax_remover,
            max_inpaint_height=self.config.video.max_inpaint_resolution,
            feather_px=self.config.mask.feather_radius,
        )

    def _report(self, callback, stage, message, **kwargs):
        if callback:
            callback(ProcessingProgress(stage=stage, message=message, **kwargs))

    def cleanup(self):
        """임시 파일을 정리한다."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
