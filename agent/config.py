"""설정 로딩 및 검증 모듈."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunPodConfig:
    pod_id: str = ""
    port: int = 8000
    api_key: str = ""
    timeout_seconds: int = 600
    health_check_interval: int = 5
    custom_url: str = ""  # 직접 URL 입력 시 pod_id/port 무시

    @property
    def base_url(self) -> str:
        """서버 base URL 반환. custom_url이 있으면 우선 사용."""
        if self.custom_url.strip():
            return self.custom_url.strip().rstrip("/")
        if not self.pod_id:
            raise ValueError("RunPod pod_id 또는 직접 URL을 입력해주세요")
        return f"https://{self.pod_id}-{self.port}.proxy.runpod.net"


@dataclass
class VideoConfig:
    max_inpaint_resolution: int = 480
    chunk_size: int = 81
    chunk_overlap: int = 11
    output_codec: str = "libx264"
    output_crf: int = 18
    output_preset: str = "medium"


@dataclass
class MiniMaxRemoverConfig:
    num_inference_steps: int = 12
    seed: int = 42
    mask_dilation: int = 6


@dataclass
class MaskConfig:
    dilation_kernel_size: int = 7
    dilation_iterations: int = 3
    feather_radius: int = 5


@dataclass
class CamPatchConfig:
    feather_radius: int = 11
    rvm_enabled: bool = False
    rvm_downsample_ratio: float = 0.25


@dataclass
class AppConfig:
    runpod: RunPodConfig = field(default_factory=RunPodConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    minimax_remover: MiniMaxRemoverConfig = field(default_factory=MiniMaxRemoverConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    campatch: CamPatchConfig = field(default_factory=CamPatchConfig)


def _find_config_path() -> Path:
    """프로젝트 루트의 config.yaml 경로를 찾는다."""
    # agent/ 디렉토리의 부모 = 프로젝트 루트
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "config.yaml"


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """YAML 파일에서 설정을 로드한다. 파일이 없으면 기본값 사용."""
    path = Path(config_path) if config_path else _find_config_path()

    config = AppConfig()

    if not path.exists():
        return config

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # runpod
    if "runpod" in data:
        for k, v in data["runpod"].items():
            if hasattr(config.runpod, k):
                setattr(config.runpod, k, v)

    # video
    if "video" in data:
        for k, v in data["video"].items():
            if hasattr(config.video, k):
                setattr(config.video, k, v)

    # minimax_remover
    if "minimax_remover" in data:
        for k, v in data["minimax_remover"].items():
            if hasattr(config.minimax_remover, k):
                setattr(config.minimax_remover, k, v)

    # mask
    if "mask" in data:
        for k, v in data["mask"].items():
            if hasattr(config.mask, k):
                setattr(config.mask, k, v)

    # campatch
    if "campatch" in data:
        for k, v in data["campatch"].items():
            if hasattr(config.campatch, k):
                setattr(config.campatch, k, v)

    _validate_config(config)
    return config


def _validate_config(config: AppConfig) -> None:
    """설정 값 검증."""
    v = config.video
    if v.chunk_overlap >= v.chunk_size:
        raise ValueError(
            f"chunk_overlap({v.chunk_overlap})이 "
            f"chunk_size({v.chunk_size})보다 작아야 합니다"
        )
    # max_inpaint_resolution을 8의 배수로 내림
    config.video.max_inpaint_resolution = (v.max_inpaint_resolution // 8) * 8
    if config.video.max_inpaint_resolution == 0:
        config.video.max_inpaint_resolution = 480
