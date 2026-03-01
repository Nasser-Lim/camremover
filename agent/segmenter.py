"""SAM2 기반 객체 세그멘테이션 모듈 — RunPod GPU 서버에서 실행.

첫 프레임에서 사용자가 클릭한 포인트를 기반으로
카메라/삼각대 등 제거 대상의 정밀 마스크를 생성한다.
"""

import io
import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw

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

    # 이미지를 PNG로 인코딩 (PIL 사용)
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")

    # API 호출
    url = f"{_server_url}/segment"
    files = {"image": ("frame.png", buf.getvalue(), "image/png")}
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

    # PNG 응답 디코딩 (PIL 사용)
    mask_img = Image.open(io.BytesIO(resp.content)).convert("L")
    mask = np.array(mask_img)

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

    # 포인트 표시 (PIL ImageDraw 사용)
    pil_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_img)

    for px, py in positive_points:
        r = 8
        draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=(0, 255, 0))
        draw.ellipse([(px - r, py - r), (px + r, py + r)], outline=(255, 255, 255), width=2)

    if negative_points:
        for px, py in negative_points:
            r = 8
            draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=(255, 0, 0))
            draw.ellipse([(px - r, py - r), (px + r, py + r)], outline=(255, 255, 255), width=2)

    return np.array(pil_img)
