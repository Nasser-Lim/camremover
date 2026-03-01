"""RunPod GPU Pod HTTP 클라이언트."""

import logging
import time
from typing import Callable, Optional

import requests

from .config import MiniMaxRemoverConfig, RunPodConfig

logger = logging.getLogger("camremover.runpod_client")


class RunPodClient:
    """RunPod Pod와 HTTP로 통신하는 클라이언트."""

    def __init__(self, config: RunPodConfig):
        self.config = config
        self.session = requests.Session()

    @property
    def base_url(self) -> str:
        return self.config.base_url

    def health_check(self) -> dict:
        """Pod 상태를 확인한다."""
        resp = self.session.get(
            f"{self.base_url}/health",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def wait_for_ready(
        self,
        max_wait_seconds: int = 300,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Pod가 준비될 때까지 폴링한다.

        Returns:
            True: 준비 완료, False: 타임아웃
        """
        start = time.time()
        interval = self.config.health_check_interval

        while time.time() - start < max_wait_seconds:
            try:
                result = self.health_check()
                if result.get("status") == "healthy":
                    if progress_callback:
                        progress_callback("GPU Pod 연결 완료")
                    return True
                if progress_callback:
                    progress_callback(
                        f"Pod 상태: {result.get('status', 'unknown')} — 모델 로딩 대기 중..."
                    )
            except requests.ConnectionError:
                if progress_callback:
                    progress_callback("Pod에 연결 중...")
            except requests.Timeout:
                if progress_callback:
                    progress_callback("Pod 응답 대기 중...")
            except Exception as e:
                if progress_callback:
                    progress_callback(f"연결 오류: {e}")

            time.sleep(interval)

        return False

    def inpaint_chunk(
        self,
        video_chunk_path: str,
        mask_path: str,
        minimax_config: MiniMaxRemoverConfig,
        max_inpaint_height: int = 480,
        feather_px: int = 5,
    ) -> bytes:
        """
        비디오 청크 + 마스크를 Pod로 보내서 인페인팅한다.

        Returns:
            인페인팅된 mp4 바이트
        """
        with open(video_chunk_path, "rb") as vf, open(mask_path, "rb") as mf:
            files = {
                "video": ("chunk.mp4", vf, "video/mp4"),
                "mask": ("mask.png", mf, "image/png"),
            }
            data = {
                "num_inference_steps": minimax_config.num_inference_steps,
                "seed": minimax_config.seed,
                "mask_dilation": minimax_config.mask_dilation,
                "max_inpaint_height": max_inpaint_height,
                "feather_px": feather_px,
            }
            resp = self.session.post(
                f"{self.base_url}/inpaint",
                files=files,
                data=data,
                timeout=self.config.timeout_seconds,
            )

        if resp.status_code != 200:
            # 서버 에러 시 응답 본문(traceback 등)을 로그에 출력
            try:
                detail = resp.text[:2000]
            except Exception:
                detail = "(응답 본문 읽기 실패)"
            logger.error(
                f"Pod 응답 에러 {resp.status_code}:\n{detail}"
            )
            resp.raise_for_status()
        return resp.content

    def rvm_matting(
        self,
        video_path: str,
        background_path: str,
        downsample_ratio: float = 0.25,
    ) -> bytes:
        """
        비디오 + 클린 레퍼런스 배경을 Pod로 보내서 RVM 알파 마스크를 받는다.

        Returns:
            알파 그레이스케일 mp4 바이트 (프레임별 전경 확률 0~255)
        """
        with open(video_path, "rb") as vf, open(background_path, "rb") as bf:
            files = {
                "video": ("video.mp4", vf, "video/mp4"),
                "background": ("background.png", bf, "image/png"),
            }
            data = {"downsample_ratio": downsample_ratio}
            resp = self.session.post(
                f"{self.base_url}/rvm_matting",
                files=files,
                data=data,
                timeout=self.config.timeout_seconds,
            )

        if resp.status_code != 200:
            try:
                detail = resp.text[:2000]
            except Exception:
                detail = "(응답 본문 읽기 실패)"
            logger.error(f"RVM matting 에러 {resp.status_code}:\n{detail}")
            resp.raise_for_status()
        return resp.content

    def unload_model(self, target: str = "all") -> dict:
        """Pod GPU 메모리를 해제한다.

        Args:
            target: "minimax" | "rvm" | "sam2" | "all"
        """
        try:
            resp = self.session.post(
                f"{self.base_url}/unload_model",
                data={"target": target},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Model unloaded: {result}")
            return result
        except Exception as e:
            logger.warning(f"unload_model 실패 (무시): {e}")
            return {}

    def reload_model(self, target: str = "minimax") -> dict:
        """Pod 모델을 다시 로드한다.

        Args:
            target: "minimax" | "rvm" | "sam2"
        """
        try:
            resp = self.session.post(
                f"{self.base_url}/reload_model",
                data={"target": target},
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"Model reloaded: {result}")
            return result
        except Exception as e:
            logger.warning(f"reload_model 실패 (무시): {e}")
            return {}

    def close(self):
        """HTTP 세션을 닫는다."""
        self.session.close()
