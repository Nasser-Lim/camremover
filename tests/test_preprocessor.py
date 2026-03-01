"""preprocessor 모듈 단위 테스트."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from agent.config import MaskConfig
from agent.preprocessor import (
    VideoChunk,
    chunk_video,
    extract_first_frame,
    get_video_info,
    preprocess_mask,
    save_mask_as_png,
)


def _create_test_video(path, num_frames=30, width=256, height=256, fps=30.0):
    """합성 테스트 영상을 생성한다."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = i * 8 % 256  # 프레임마다 다른 색상
        cv2.putText(
            frame, str(i), (10, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
        )
        writer.write(frame)
    writer.release()


class TestExtractFirstFrame:
    def test_returns_correct_shape(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=10, width=320, height=240)

        frame = extract_first_frame(video_path)
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError):
            extract_first_frame("/nonexistent/video.mp4")


class TestGetVideoInfo:
    def test_returns_correct_metadata(self, tmp_path):
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=60, width=640, height=480, fps=30.0)

        info = get_video_info(video_path)
        assert info.width == 640
        assert info.height == 480
        assert info.total_frames == 60
        assert abs(info.fps - 30.0) < 1.0
        assert info.duration_seconds > 0


class TestPreprocessMask:
    def test_rgba_mask_extraction(self):
        """RGBA 마스크에서 alpha 채널 기반 추출."""
        raw = np.zeros((100, 100, 4), dtype=np.uint8)
        raw[30:60, 30:60, 3] = 255  # alpha 채널에 사각형

        config = MaskConfig(dilation_kernel_size=0, dilation_iterations=0)
        result = preprocess_mask(raw, target_size=(100, 100), config=config)

        assert result.shape == (100, 100)
        assert result[45, 45] == 255  # 사각형 내부
        assert result[0, 0] == 0  # 사각형 외부

    def test_dilation_enlarges_mask(self):
        """팽창이 마스크를 확장하는지 확인."""
        raw = np.zeros((100, 100, 4), dtype=np.uint8)
        raw[40:60, 40:60, 3] = 255

        config_no_dilation = MaskConfig(dilation_kernel_size=0, dilation_iterations=0)
        config_dilation = MaskConfig(dilation_kernel_size=5, dilation_iterations=2)

        mask_no = preprocess_mask(raw, (100, 100), config_no_dilation)
        mask_yes = preprocess_mask(raw, (100, 100), config_dilation)

        # 팽창 후 더 많은 흰색 픽셀
        assert mask_yes.sum() > mask_no.sum()

    def test_resize_to_target(self):
        """마스크를 영상 크기로 리사이즈."""
        raw = np.zeros((50, 50, 4), dtype=np.uint8)
        raw[10:40, 10:40, 3] = 255

        config = MaskConfig(dilation_kernel_size=0, dilation_iterations=0)
        result = preprocess_mask(raw, target_size=(200, 200), config=config)

        assert result.shape == (200, 200)


class TestChunkVideo:
    def test_correct_chunk_count(self, tmp_path):
        """250프레임, chunk=81, overlap=11 → 4 청크."""
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=250, width=128, height=128)

        output_dir = str(tmp_path / "chunks")
        chunks = chunk_video(video_path, chunk_size=81, overlap=11, output_dir=output_dir)

        # stride=70, (250-81)/70 + 1 ≈ 3.4 → 4 chunks
        assert len(chunks) >= 3

        # 모든 프레임이 커버되는지 확인
        covered = set()
        for c in chunks:
            for f in range(c.frame_start, c.frame_end):
                covered.add(f)
        assert len(covered) == 250

    def test_short_video_single_chunk(self, tmp_path):
        """프레임 수 < chunk_size → 단일 청크 (패딩 적용)."""
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=30, width=128, height=128)

        output_dir = str(tmp_path / "chunks")
        chunks = chunk_video(video_path, chunk_size=81, overlap=11, output_dir=output_dir)

        assert len(chunks) == 1
        assert chunks[0].overlap_start == 0
        assert chunks[0].overlap_end == 0
        assert chunks[0].padded_frames == 81 - 30  # 51프레임 패딩

    def test_overlap_metadata(self, tmp_path):
        """첫 청크는 overlap_start=0, 마지막 청크는 overlap_end=0."""
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=250, width=128, height=128)

        output_dir = str(tmp_path / "chunks")
        chunks = chunk_video(video_path, chunk_size=81, overlap=11, output_dir=output_dir)

        assert chunks[0].overlap_start == 0
        assert chunks[-1].overlap_end == 0

        # 중간 청크는 양쪽 overlap
        if len(chunks) > 2:
            assert chunks[1].overlap_start == 11

    def test_chunk_files_exist(self, tmp_path):
        """청크 mp4 파일이 실제로 생성되는지."""
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=200, width=128, height=128)

        output_dir = str(tmp_path / "chunks")
        chunks = chunk_video(video_path, chunk_size=81, overlap=11, output_dir=output_dir)

        for c in chunks:
            assert os.path.exists(c.file_path)
            assert os.path.getsize(c.file_path) > 0

    def test_last_chunk_padding(self, tmp_path):
        """마지막 청크가 chunk_size 미만이면 패딩 적용."""
        video_path = str(tmp_path / "test.mp4")
        _create_test_video(video_path, num_frames=100, width=128, height=128)

        output_dir = str(tmp_path / "chunks")
        chunks = chunk_video(video_path, chunk_size=81, overlap=11, output_dir=output_dir)

        # 마지막 청크의 패딩 확인
        last = chunks[-1]
        actual_frames = last.frame_end - last.frame_start
        if actual_frames < 81:
            assert last.padded_frames == 81 - actual_frames
        else:
            assert last.padded_frames == 0

        # 패딩 없는 청크는 padded_frames=0
        if len(chunks) > 1:
            assert chunks[0].padded_frames == 0


class TestSaveMaskAsPng:
    def test_save_and_reload(self, tmp_path):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        path = str(tmp_path / "mask.png")
        save_mask_as_png(mask, path)

        loaded = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert loaded is not None
        assert loaded.shape == (100, 100)
        assert loaded[50, 50] == 255
        assert loaded[0, 0] == 0
