"""postprocessor 모듈 단위 테스트."""

import os

import cv2
import numpy as np
import pytest

from agent.config import VideoConfig
from agent.postprocessor import _crossfade_frames, merge_chunks
from agent.preprocessor import VideoChunk, VideoInfo


def _create_test_video(path, num_frames=10, width=128, height=128, fps=30.0, color=(0, 0, 0)):
    """단색 합성 영상을 생성한다."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestCrossfadeFrames:
    def test_linear_blend(self):
        """흑→백 크로스페이드가 선형 그라데이션을 만드는지."""
        n = 10
        black = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(n)]
        white = [np.full((64, 64, 3), 255, dtype=np.uint8) for _ in range(n)]

        blended = _crossfade_frames(black, white, n)

        assert len(blended) == n

        # 첫 프레임은 거의 검정, 마지막은 거의 흰색
        assert blended[0].mean() < 20
        assert blended[-1].mean() > 230

        # 중간은 대략 128 부근
        mid_val = blended[n // 2].mean()
        assert 100 < mid_val < 160

    def test_single_frame_overlap(self):
        """겹침 1프레임에서도 동작."""
        a = [np.zeros((32, 32, 3), dtype=np.uint8)]
        b = [np.full((32, 32, 3), 200, dtype=np.uint8)]

        blended = _crossfade_frames(a, b, 1)
        assert len(blended) == 1
        # alpha = 0.5 → 100
        assert 80 < blended[0].mean() < 120


class TestMergeChunks:
    def test_single_chunk(self, tmp_path):
        """단일 청크 merge는 동일 결과."""
        video_path = str(tmp_path / "chunk.mp4")
        _create_test_video(video_path, num_frames=20, width=128, height=128, color=(100, 100, 100))

        video_info = VideoInfo(
            width=128, height=128, fps=30.0,
            total_frames=20, duration_seconds=0.67,
            has_audio=False, file_path="",
        )

        chunk = VideoChunk(
            chunk_index=0, frame_start=0, frame_end=20,
            overlap_start=0, overlap_end=0,
            file_path=video_path, total_frames=20,
        )

        output_path = str(tmp_path / "merged.mp4")
        config = VideoConfig()
        merge_chunks([{"chunk": chunk, "result_path": video_path}], video_info, output_path, config)

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_two_chunks_with_overlap(self, tmp_path):
        """2개 청크를 겹침으로 merge."""
        v1_path = str(tmp_path / "chunk1.mp4")
        v2_path = str(tmp_path / "chunk2.mp4")
        _create_test_video(v1_path, num_frames=20, width=128, height=128, color=(50, 50, 50))
        _create_test_video(v2_path, num_frames=20, width=128, height=128, color=(200, 200, 200))

        video_info = VideoInfo(
            width=128, height=128, fps=30.0,
            total_frames=30, duration_seconds=1.0,
            has_audio=False, file_path="",
        )

        chunk1 = VideoChunk(
            chunk_index=0, frame_start=0, frame_end=20,
            overlap_start=0, overlap_end=10,
            file_path=v1_path, total_frames=20,
        )
        chunk2 = VideoChunk(
            chunk_index=1, frame_start=10, frame_end=30,
            overlap_start=10, overlap_end=0,
            file_path=v2_path, total_frames=20,
        )

        output_path = str(tmp_path / "merged.mp4")
        config = VideoConfig()
        merge_chunks(
            [
                {"chunk": chunk1, "result_path": v1_path},
                {"chunk": chunk2, "result_path": v2_path},
            ],
            video_info,
            output_path,
            config,
        )

        assert os.path.exists(output_path)

        # 결과 프레임 수: 20 + (20-10) = 30
        cap = cv2.VideoCapture(output_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert frame_count == 30

    def test_padded_chunk_trimmed(self, tmp_path):
        """패딩된 마지막 청크에서 패딩 프레임이 제거되는지 확인."""
        video_path = str(tmp_path / "chunk.mp4")
        # 실제로는 81프레임 (30프레임 실제 + 51프레임 패딩)
        _create_test_video(video_path, num_frames=81, width=128, height=128, color=(100, 100, 100))

        video_info = VideoInfo(
            width=128, height=128, fps=30.0,
            total_frames=30, duration_seconds=1.0,
            has_audio=False, file_path="",
        )

        chunk = VideoChunk(
            chunk_index=0, frame_start=0, frame_end=30,
            overlap_start=0, overlap_end=0,
            file_path=video_path, total_frames=81,
            padded_frames=51,
        )

        output_path = str(tmp_path / "merged.mp4")
        config = VideoConfig()
        merge_chunks([{"chunk": chunk, "result_path": video_path}], video_info, output_path, config)

        assert os.path.exists(output_path)
        cap = cv2.VideoCapture(output_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # 81프레임 중 51프레임 패딩 제거 → 30프레임
        assert frame_count == 30
