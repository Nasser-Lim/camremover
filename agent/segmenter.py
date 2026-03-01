"""SAM2 기반 객체 세그멘테이션 모듈 — RunPod GPU 서버에서 실행.

첫 프레임에서 사용자가 클릭한 포인트를 기반으로
카메라/삼각대 등 제거 대상의 정밀 마스크를 생성한다.
"""

import json
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger("camremover.segmenter")

# RunPod 서버 URL (set_server_url()로 설정)
_server_url: str | None = None


def set_server_url(url: str):
    """RunPod 서버 URL을 설정한다."""
    global _server_url
    _server_url = url.rstrip("/")
    logger.info(f"Segmenter server URL: {_server_url}")


def segment_from_points(
    image_rgb: np.ndarray,
    positive_points: List[Tuple[int, int]],
    negative_points: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    RunPod 서버의 SAM2로 객체를 세그멘테이션한다.

    Args:
        image_rgb: HxWx3 uint8 RGB 이미지
        positive_points: [(x, y), ...] — 제거 대상 위의 클릭 좌표
        negative_points: [(x, y), ...] — 보존 대상 위의 클릭 좌표 (선택)

    Returns:
        HxW uint8 바이너리 마스크 (255=제거, 0=유지)
    """
    if not positive_points:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    if _server_url is None:
        raise RuntimeError("서버 URL이 설정되지 않았습니다. set_server_url()을 먼저 호출하세요.")

    # 이미지를 PNG로 인코딩
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, img_png = cv2.imencode(".png", img_bgr)

    # API 호출
    url = f"{_server_url}/segment"
    files = {"image": ("frame.png", img_png.tobytes(), "image/png")}
    data = {
        "positive_points": json.dumps([list(p) for p in positive_points]),
        "negative_points": json.dumps([list(p) for p in (negative_points or [])]),
    }

    logger.info(
        f"SAM2 request: {len(positive_points)} pos, "
        f"{len(negative_points or [])} neg points → {url}"
    )

    resp = requests.post(url, files=files, data=data, timeout=60)
    if resp.status_code != 200:
        detail = resp.text[:500]
        logger.error(f"SAM2 server error {resp.status_code}: {detail}")
        raise RuntimeError(f"SAM2 서버 에러 {resp.status_code}: {detail}")

    # PNG 응답 디코딩
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("SAM2 서버 응답에서 마스크를 디코딩할 수 없습니다")

    logger.info(
        f"SAM2 result: mask shape={mask.shape}, "
        f"coverage={np.count_nonzero(mask > 127) / mask.size:.1%}"
    )

    return mask


def create_mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    positive_points: List[Tuple[int, int]],
    negative_points: Optional[List[Tuple[int, int]]] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    마스크 오버레이 프리뷰를 생성한다.

    마스크 영역을 반투명 빨간색으로, positive 포인트를 초록, negative를 빨강으로 표시.

    Returns:
        HxWx3 uint8 RGB 이미지
    """
    overlay = image_rgb.copy()

    # 마스크 영역을 빨간색으로
    if mask is not None and mask.max() > 0:
        mask_bool = mask > 127
        red = np.array([255, 60, 60], dtype=np.uint8)
        overlay[mask_bool] = (
            overlay[mask_bool].astype(np.float32) * (1 - alpha)
            + red.astype(np.float32) * alpha
        ).astype(np.uint8)

    # 포인트 표시
    for px, py in positive_points:
        cv2.circle(overlay, (int(px), int(py)), 8, (0, 255, 0), -1)
        cv2.circle(overlay, (int(px), int(py)), 8, (255, 255, 255), 2)

    if negative_points:
        for px, py in negative_points:
            cv2.circle(overlay, (int(px), int(py)), 8, (255, 0, 0), -1)
            cv2.circle(overlay, (int(px), int(py)), 8, (255, 255, 255), 2)

    return overlay
