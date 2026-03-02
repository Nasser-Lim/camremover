"""Gradio 웹 UI — SAM2 클릭 세그멘테이션 + 브러시 마스크, 영상 처리, 결과 다운로드."""

import logging
import os
import subprocess
import tempfile
from typing import Optional

import cv2
import gradio as gr
import numpy as np

from .config import AppConfig, load_config
from .main import CamRemoverAgent, ProcessingProgress
from .preprocessor import extract_first_frame, extract_frame_at, get_video_info

logger = logging.getLogger("camremover.ui")


def _to_browser_mp4(src: str) -> str:
    """
    브라우저 재생 가능한 mp4를 반환한다.
    - 이미 .mp4이면 그대로 반환
    - .mov 등이면 임시 mp4로 변환 (H.264/H.265는 -c copy, 그 외 재인코딩)
    """
    ext = os.path.splitext(src)[1].lower()
    if ext == ".mp4":
        return src

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1:nokey=1", src],
        capture_output=True, text=True, timeout=10,
    )
    codec = probe.stdout.strip().lower()

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    if codec in ("h264", "hevc"):
        vargs = ["-c:v", "copy", "-c:a", "copy"]
    else:
        vargs = [
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        ]

    subprocess.run(
        ["ffmpeg", "-y", "-i", src] + vargs + [tmp.name],
        capture_output=True, timeout=300, check=True,
    )
    return tmp.name


CUSTOM_CSS = """
/* ── 전체 레이아웃 ── */
.main-title { text-align: center; margin-bottom: 0.2em; }
.main-title h1 { font-size: 1.8em; margin: 0; }
.main-subtitle {
    text-align: center; color: #666; margin-bottom: 1.5em;
    font-size: 0.9em; line-height: 1.5;
}

/* ── 섹션 헤더 ── */
.section-header {
    font-size: 1.05em; font-weight: 600;
    border-left: 4px solid #6366f1;
    padding-left: 10px; margin-bottom: 0.5em;
    color: #1f2937;
}

/* ── Pod 연결 상태 ── */
.pod-status-connected {
    background: #d1fae5; border: 1px solid #6ee7b7;
    border-radius: 8px; padding: 8px 14px;
    color: #065f46; font-weight: 600; font-size: 0.92em;
}
.pod-status-disconnected {
    background: #fee2e2; border: 1px solid #fca5a5;
    border-radius: 8px; padding: 8px 14px;
    color: #991b1b; font-weight: 600; font-size: 0.92em;
}
.pod-status-checking {
    background: #fef9c3; border: 1px solid #fde047;
    border-radius: 8px; padding: 8px 14px;
    color: #713f12; font-weight: 600; font-size: 0.92em;
}

/* ── 처리 버튼 ── */
.process-btn { min-height: 52px !important; font-size: 1.1em !important; }

/* ── 상태창 ── */
.status-box textarea { font-size: 0.92em !important; }

/* ── 포인트 범례 ── */
.point-legend {
    font-size: 0.87em; color: #555; line-height: 1.8;
    padding: 8px 12px; background: #f8f9fa;
    border-radius: 6px; margin-bottom: 6px;
}

/* ── RunPod 안내 배너 ── */
.pod-info-banner {
    background: #eff6ff; border: 1px solid #93c5fd;
    border-radius: 8px; padding: 10px 14px;
    color: #1e40af; font-size: 0.88em; text-align: center;
    margin: 4px 0; font-weight: 500;
}

/* ── SAM2 잠금 경고 배너 ── */
.sam-locked-banner {
    background: #fef3c7; border: 1px solid #f59e0b;
    border-radius: 8px; padding: 10px 14px;
    color: #92400e; font-size: 0.88em; text-align: center;
    margin: 4px 0; font-weight: 600;
}
"""


def create_ui(config: Optional[AppConfig] = None) -> gr.Blocks:
    """Gradio Blocks 앱을 빌드하여 반환한다."""
    config = config or load_config()

    # SAM2 서버 URL 설정 (RunPod Pod) — pod_id 없으면 빈 URL로 초기화
    from .segmenter import set_server_url
    try:
        set_server_url(config.runpod.base_url)
    except ValueError:
        set_server_url("")

    with gr.Blocks(title="SBS Camremover v1.0.0") as app:

        # ── 헤더 ──
        gr.HTML(
            '<div class="main-title"><h1>SBS Camremover v1.0.0</h1></div>'
            '<div class="main-subtitle">'
            '영상 속 거치카메라 · 삼각대를 AI로 자동 제거합니다.<br>'
            'SAM2 클릭으로 정밀 선택하거나, 브러시로 직접 칠할 수 있습니다.'
            '</div>'
        )

        # ── 상태 저장 ──
        state_config = gr.State(value=config)
        state_video_path = gr.State(value=None)
        state_first_frame = gr.State(value=None)
        state_positive_pts = gr.State(value=[])
        state_negative_pts = gr.State(value=[])
        state_sam_mask = gr.State(value=None)
        state_brush_mask = gr.State(value=None)
        state_pod_connected = gr.State(value=False)

        # ════════════════════════════════════════
        #  RunPod 연결 (Accordion, 기본 접힘)
        # ════════════════════════════════════════
        with gr.Accordion("🔌 RunPod 연결 설정", open=False):
            gr.HTML(
                '<div class="pod-info-banner">'
                '🖥️ SAM2 클릭 마스크 · MiniMax-Remover · RVM 피사체 보호 기능에 RunPod GPU 서버가 필요합니다.'
                '</div>'
            )
            with gr.Row():
                pod_id_input = gr.Textbox(
                    label="Pod ID",
                    value=config.runpod.pod_id,
                    placeholder="예: abc123xyz",
                    info="RunPod 대시보드에서 확인",
                    scale=2,
                )
                pod_port_input = gr.Number(
                    label="포트",
                    value=config.runpod.port,
                    precision=0,
                    minimum=1,
                    maximum=65535,
                    scale=1,
                )
            pod_url_input = gr.Textbox(
                label="직접 URL (선택 — 입력 시 Pod ID/포트 무시)",
                placeholder="예: https://abc123xyz-8000.proxy.runpod.net  또는  http://내서버:8000",
            )
            with gr.Row():
                pod_connect_btn = gr.Button("연결 테스트", variant="secondary", scale=1)
                pod_status_html = gr.HTML(
                    '<div class="pod-status-disconnected">⚠ 미연결 — GPU 기능 비활성화됨</div>',
                    scale=3,
                )

        # ════════════════════════════════════════
        #  Step 1: 영상 업로드
        # ════════════════════════════════════════
        with gr.Group():
            gr.HTML('<div class="section-header">Step 1 &nbsp; 영상 업로드</div>')
            video_input = gr.File(
                label="영상 파일 (.mp4 / .mov / .avi)",
                file_types=["video"],
            )

        # ════════════════════════════════════════
        #  Step 2: 마스크 생성
        # ════════════════════════════════════════
        with gr.Group():
            gr.HTML('<div class="section-header">Step 2 &nbsp; 제거 대상 선택</div>')

            frame_slider = gr.Slider(
                label="기준 프레임",
                minimum=0, maximum=1, step=1, value=0,
                info="카메라/삼각대가 가장 잘 보이는 프레임을 선택하세요",
                interactive=True,
            )

            mask_mode = gr.Radio(
                choices=["SAM 클릭 (권장)", "브러시"],
                value="브러시",
                label="마스크 모드",
                interactive=True,
            )

            # ── SAM 클릭 모드 ──
            with gr.Group(visible=False) as sam_group:
                gr.HTML(
                    '<div class="point-legend">'
                    '이미지를 클릭하면 포인트가 추가됩니다. &nbsp;'
                    '<span style="color:#16a34a;font-weight:bold">■ 제거 대상</span>'
                    ' — 카메라/삼각대 위 클릭 &nbsp;&nbsp;'
                    '<span style="color:#dc2626;font-weight:bold">■ 보존 대상</span>'
                    ' — 사람 등 지우면 안 되는 영역 클릭'
                    '</div>'
                )
                point_type = gr.Radio(
                    choices=["제거 대상 (초록)", "보존 대상 (빨강)"],
                    value="제거 대상 (초록)",
                    label="클릭 타입",
                    interactive=False,  # Pod 연결 전 비활성화
                )

                with gr.Group(visible=True) as sam_locked_banner:
                    gr.HTML(
                        '<div class="sam-locked-banner">'
                        '🔒 SAM2 클릭은 RunPod 연결이 필요합니다. &nbsp;'
                        '상단 <b>RunPod 연결 설정</b>을 열고 연결 테스트를 완료하세요.'
                        '</div>'
                    )

                sam_preview = gr.Image(
                    label="SAM2 마스크 프리뷰 (클릭하여 포인트 추가)",
                    type="numpy",
                    interactive=False,  # Pod 연결 전 비활성화
                    height=480,
                )

                with gr.Row():
                    undo_btn = gr.Button("↩ 마지막 포인트 취소", size="sm", interactive=False)
                    clear_pts_btn = gr.Button("✕ 모든 포인트 초기화", size="sm", interactive=False)

            # ── 브러시 모드 ──
            with gr.Group(visible=True) as brush_group:
                gr.Markdown(
                    "빨간 브러시로 제거할 카메라 영역을 칠하세요. 넉넉하게 칠해도 괜찮습니다."
                )
                mask_editor = gr.ImageEditor(
                    label="브러시 마스크",
                    type="numpy",
                    brush=gr.Brush(
                        colors=["#FF0000"],
                        default_size=20,
                        color_mode="fixed",
                    ),
                    eraser=gr.Eraser(default_size=20),
                    sources=[],
                    image_mode="RGBA",
                    interactive=True,
                    height=480,
                )

        # ════════════════════════════════════════
        #  Step 3: 인페인팅 설정 + 처리
        # ════════════════════════════════════════
        with gr.Group():
            gr.HTML('<div class="section-header">Step 3 &nbsp; 인페인팅 설정</div>')

            inpaint_mode = gr.Radio(
                choices=["MiniMax-Remover (GPU 서버)", "Simple LaMa (로컬)"],
                value="Simple LaMa (로컬)",
                label="인페인팅 엔진",
                interactive=True,
            )

            # ── MiniMax-Remover 전용 설정 ──
            with gr.Group(visible=False) as minimax_settings_group:
                with gr.Accordion("고급 설정 (MiniMax-Remover)", open=False):
                    with gr.Row():
                        max_res_input = gr.Slider(
                            label="최대 처리 해상도 (높이 px)",
                            minimum=360, maximum=720, step=8,
                            value=config.video.max_inpaint_resolution,
                            interactive=False,  # Pod 연결 전 비활성화
                        )
                        num_steps_input = gr.Slider(
                            label="추론 스텝 (높을수록 고품질)",
                            minimum=6, maximum=12, step=1,
                            value=config.minimax_remover.num_inference_steps,
                            interactive=False,
                        )
                    with gr.Row():
                        seed_input = gr.Number(
                            label="랜덤 시드",
                            value=config.minimax_remover.seed,
                            precision=0,
                            interactive=False,
                        )
                        mask_dilation_input = gr.Slider(
                            label="마스크 팽창 (px)",
                            minimum=0, maximum=20, step=1,
                            value=config.mask.dilation_kernel_size,
                            interactive=False,
                        )

            # ── CamPatch / Simple LaMa 설정 ──
            with gr.Group(visible=True) as campatch_settings_group:
                gr.Markdown(
                    "LaMa로 클린 배경 1장을 생성한 후 전 프레임의 마스크 영역에 합성합니다. "
                    "GPU 서버 없이 로컬에서 동작하며 프레임 간 일관성이 뛰어납니다."
                )
                campatch_feather = gr.Slider(
                    label="페더링 반경 (px)",
                    minimum=0, maximum=60, step=1,
                    value=config.campatch.feather_radius,
                    info="경계 블렌딩 부드러움 + LaMa 인페인팅 확장 범위 (0 = 하드 컷)",
                )
                with gr.Row():
                    campatch_rvm_enabled = gr.Checkbox(
                        label="RVM 피사체 보호 (RunPod Pod 필요)",
                        value=config.campatch.rvm_enabled,
                        info="RobustVideoMatting으로 전경/배경을 정밀 분리합니다",
                        interactive=False,  # Pod 연결 전 비활성화
                    )
                    campatch_rvm_ratio = gr.Slider(
                        label="RVM 다운샘플 비율",
                        minimum=0.1, maximum=1.0, step=0.05,
                        value=config.campatch.rvm_downsample_ratio,
                        info="낮을수록 빠름 (0.25 권장)",
                        interactive=False,
                    )
                with gr.Row():
                    campatch_preview_btn = gr.Button(
                        "클린 레퍼런스 미리보기", variant="secondary", size="sm",
                    )
                campatch_preview_img = gr.Image(
                    label="클린 레퍼런스 (LaMa 인페인팅 결과)",
                    interactive=False,
                    visible=False,
                )

        # ════════════════════════════════════════
        #  처리 시작
        # ════════════════════════════════════════
        with gr.Group():
            process_btn = gr.Button(
                "▶  처리 시작",
                variant="primary",
                size="lg",
                elem_classes=["process-btn"],
            )
            progress_text = gr.Textbox(
                label="진행 상태",
                interactive=False,
                value="대기 중",
                elem_classes=["status-box"],
            )

        # ════════════════════════════════════════
        #  결과
        # ════════════════════════════════════════
        with gr.Group():
            gr.HTML('<div class="section-header">결과</div>')
            result_video = gr.Video(label="인페인팅 결과", interactive=False)

        # ════════════════════════════════════════
        #  헬퍼 함수
        # ════════════════════════════════════════

        def _needs_pod(inpaint_mode_str, rvm_on):
            """처리 실행 시 Pod 연결이 필요한지 판단한다.
            SAM2 클릭은 이미 마스크 생성 시점에 Pod를 사용하므로 여기선 제외."""
            return "MiniMax" in inpaint_mode_str or rvm_on

        def _pod_url_from_inputs(pod_url, pod_id, pod_port) -> str:
            url_clean = (pod_url or "").strip().rstrip("/")
            if url_clean:
                return url_clean
            id_clean = (pod_id or "").strip()
            if id_clean:
                return f"https://{id_clean}-{int(pod_port or 8000)}.proxy.runpod.net"
            return ""

        # ════════════════════════════════════════
        #  이벤트 핸들러
        # ════════════════════════════════════════

        # ── Pod 연결 테스트 ──
        def on_pod_connect(pod_url, pod_id, pod_port):
            import requests as _req
            url = _pod_url_from_inputs(pod_url, pod_id, pod_port)
            if not url:
                return (
                    '<div class="pod-status-disconnected">⚠ Pod ID 또는 URL을 입력해주세요</div>',
                    False,
                    gr.update(visible=True),   # sam_locked_banner 표시
                    # SAM 컨트롤
                    gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False), gr.update(interactive=False),
                    # MiniMax 슬라이더들
                    gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False), gr.update(interactive=False),
                    # RVM 컨트롤
                    gr.update(interactive=False), gr.update(interactive=False),
                )
            try:
                resp = _req.get(f"{url}/health", timeout=8)
                ok = resp.status_code == 200
            except Exception:
                ok = False

            if ok:
                from .segmenter import set_server_url
                set_server_url(url)
                status_html = (
                    f'<div class="pod-status-connected">'
                    f'✅ 연결됨 — {url}'
                    f'</div>'
                )
            else:
                status_html = (
                    f'<div class="pod-status-disconnected">'
                    f'✗ 연결 실패 — {url}'
                    f'</div>'
                )

            return (
                status_html,
                ok,
                gr.update(visible=not ok),  # sam_locked_banner: 연결 성공 시 숨김
                # SAM 컨트롤 (point_type, sam_preview, undo_btn, clear_pts_btn)
                gr.update(interactive=ok), gr.update(interactive=ok),
                gr.update(interactive=ok), gr.update(interactive=ok),
                # MiniMax 슬라이더들 (max_res, num_steps, seed, mask_dilation)
                gr.update(interactive=ok), gr.update(interactive=ok),
                gr.update(interactive=ok), gr.update(interactive=ok),
                # RVM 컨트롤 (rvm_enabled, rvm_ratio)
                gr.update(interactive=ok), gr.update(interactive=ok),
            )

        pod_connect_btn.click(
            fn=on_pod_connect,
            inputs=[pod_url_input, pod_id_input, pod_port_input],
            outputs=[
                pod_status_html, state_pod_connected,
                sam_locked_banner,
                point_type, sam_preview, undo_btn, clear_pts_btn,
                max_res_input, num_steps_input, seed_input, mask_dilation_input,
                campatch_rvm_enabled, campatch_rvm_ratio,
            ],
        )

        # ── 마스크 모드 전환 ──
        def on_mode_change(mask_mode_str):
            is_sam = "SAM" in mask_mode_str
            return (
                gr.update(visible=is_sam),
                gr.update(visible=not is_sam),
            )

        mask_mode.change(
            fn=on_mode_change,
            inputs=[mask_mode],
            outputs=[sam_group, brush_group],
        )

        # ── 브러시 편집 → state 동기화 ──
        def on_brush_change(editor_value):
            return editor_value

        mask_editor.change(
            fn=on_brush_change,
            inputs=[mask_editor],
            outputs=[state_brush_mask],
        )

        # ── 인페인팅 모드 전환 ──
        def on_inpaint_mode_change(inpaint_mode_str, pod_connected):
            is_minimax = "MiniMax" in inpaint_mode_str
            # MiniMax 모드인데 Pod 미연결이면 Simple LaMa 설정 패널만 보여줌
            # (inpaint_mode 자체를 outputs에 넣으면 무한루프 발생 → 패널 전환만 처리)
            if is_minimax and not pod_connected:
                return (
                    gr.update(visible=False),  # minimax_settings
                    gr.update(visible=True),   # campatch_settings
                )
            return (
                gr.update(visible=is_minimax),
                gr.update(visible=not is_minimax),
            )

        inpaint_mode.change(
            fn=on_inpaint_mode_change,
            inputs=[inpaint_mode, state_pod_connected],
            outputs=[minimax_settings_group, campatch_settings_group],
        )

        # ── CamPatch 미리보기 ──
        def on_campatch_preview(
            video_path, mask_mode_str, sam_mask, editor_value,
            campatch_feather_val, frame_idx, cfg,
        ):
            if video_path is None:
                return gr.update(visible=False), "오류: 영상이 없습니다"
            if "SAM" in mask_mode_str:
                mask_raw = sam_mask
            else:
                layers = (editor_value or {}).get("layers", [])
                mask_raw = layers[0] if layers else None
            if mask_raw is None:
                return gr.update(visible=False), "오류: 마스크가 없습니다"

            cfg.campatch.feather_radius = int(campatch_feather_val)
            from .campatch import generate_clean_reference
            try:
                video_path = _to_browser_mp4(video_path)
                ref_img = generate_clean_reference(
                    video_path, mask_raw, cfg, ref_frame_idx=int(frame_idx)
                )
                return gr.update(value=ref_img, visible=True), "클린 레퍼런스 생성 완료"
            except Exception as e:
                logger.exception("클린 레퍼런스 생성 실패")
                return gr.update(visible=False), f"오류: {e}"

        campatch_preview_btn.click(
            fn=on_campatch_preview,
            inputs=[
                state_video_path, mask_mode,
                state_sam_mask, state_brush_mask,
                campatch_feather,
                frame_slider, state_config,
            ],
            outputs=[campatch_preview_img, progress_text],
        )

        # ── 영상 업로드 ──
        def on_video_upload(file_obj):
            if file_obj is None:
                return (
                    gr.update(value=None), gr.update(value=None),
                    gr.update(maximum=1, value=0),
                    None, None, [], [], None, None,
                )

            video_path = file_obj if isinstance(file_obj, str) else file_obj.name

            try:
                video_info = get_video_info(video_path)
                total = max(video_info.total_frames - 1, 1)

                first_frame = extract_first_frame(video_path)
                first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

                editor_value = {
                    "background": first_frame_rgb,
                    "layers": [],
                    "composite": first_frame_rgb,
                }

                return (
                    gr.update(value=first_frame_rgb),
                    gr.update(value=editor_value),
                    gr.update(maximum=total, value=0),
                    video_path,
                    first_frame_rgb,
                    [], [], None, None,
                )
            except Exception as e:
                logger.error(f"첫 프레임 추출 실패: {e}")
                return (
                    gr.update(value=None), gr.update(value=None),
                    gr.update(maximum=1, value=0),
                    None, None, [], [], None, None,
                )

        video_input.change(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[
                sam_preview, mask_editor,
                frame_slider,
                state_video_path, state_first_frame,
                state_positive_pts, state_negative_pts, state_sam_mask,
                state_brush_mask,
            ],
        )

        # ── 프레임 슬라이더 ──
        def on_frame_change(frame_idx, video_path):
            if video_path is None:
                return gr.update(), gr.update(), None, [], [], None, None

            try:
                frame_bgr = extract_frame_at(video_path, int(frame_idx))
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                editor_value = {
                    "background": frame_rgb,
                    "layers": [],
                    "composite": frame_rgb,
                }

                return (
                    gr.update(value=frame_rgb),
                    gr.update(value=editor_value),
                    frame_rgb,
                    [], [], None, None,
                )
            except Exception as e:
                logger.error(f"프레임 {frame_idx} 추출 실패: {e}")
                return gr.update(), gr.update(), None, [], [], None, None

        frame_slider.release(
            fn=on_frame_change,
            inputs=[frame_slider, state_video_path],
            outputs=[
                sam_preview, mask_editor,
                state_first_frame,
                state_positive_pts, state_negative_pts, state_sam_mask,
                state_brush_mask,
            ],
        )

        # ── SAM2 클릭 ──
        def on_sam_click(
            point_type_str,
            first_frame, pos_pts, neg_pts,
            pod_url, pod_id, pod_port,
            evt: gr.SelectData,
        ):
            if first_frame is None:
                return gr.update(), pos_pts, neg_pts, None

            from .segmenter import create_mask_overlay, segment_from_points, set_server_url

            url = _pod_url_from_inputs(pod_url, pod_id, pod_port)
            if url:
                set_server_url(url)

            x, y = evt.index
            pos_pts = list(pos_pts)
            neg_pts = list(neg_pts)

            if "제거" in point_type_str:
                pos_pts.append((x, y))
            else:
                neg_pts.append((x, y))

            mask = segment_from_points(
                first_frame,
                positive_points=pos_pts,
                negative_points=neg_pts if neg_pts else None,
            )

            overlay = create_mask_overlay(first_frame, mask, pos_pts, neg_pts)
            return gr.update(value=overlay), pos_pts, neg_pts, mask

        sam_preview.select(
            fn=on_sam_click,
            inputs=[
                point_type,
                state_first_frame, state_positive_pts, state_negative_pts,
                pod_url_input, pod_id_input, pod_port_input,
            ],
            outputs=[
                sam_preview,
                state_positive_pts, state_negative_pts, state_sam_mask,
            ],
        )

        # ── SAM2 포인트 취소 ──
        def on_undo(first_frame, pos_pts, neg_pts, pod_url, pod_id, pod_port):
            if first_frame is None:
                return gr.update(), pos_pts, neg_pts, None

            pos_pts = list(pos_pts)
            neg_pts = list(neg_pts)

            if neg_pts:
                neg_pts.pop()
            elif pos_pts:
                pos_pts.pop()
            else:
                return gr.update(value=first_frame), pos_pts, neg_pts, None

            if not pos_pts and not neg_pts:
                return gr.update(value=first_frame), pos_pts, neg_pts, None

            from .segmenter import create_mask_overlay, segment_from_points, set_server_url

            url = _pod_url_from_inputs(pod_url, pod_id, pod_port)
            if url:
                set_server_url(url)

            mask = segment_from_points(
                first_frame,
                positive_points=pos_pts,
                negative_points=neg_pts if neg_pts else None,
            )
            overlay = create_mask_overlay(first_frame, mask, pos_pts, neg_pts)
            return gr.update(value=overlay), pos_pts, neg_pts, mask

        undo_btn.click(
            fn=on_undo,
            inputs=[
                state_first_frame, state_positive_pts, state_negative_pts,
                pod_url_input, pod_id_input, pod_port_input,
            ],
            outputs=[
                sam_preview,
                state_positive_pts, state_negative_pts, state_sam_mask,
            ],
        )

        # ── SAM2 초기화 ──
        def on_clear(first_frame):
            if first_frame is None:
                return gr.update(), [], [], None
            return gr.update(value=first_frame), [], [], None

        clear_pts_btn.click(
            fn=on_clear,
            inputs=[state_first_frame],
            outputs=[
                sam_preview,
                state_positive_pts, state_negative_pts, state_sam_mask,
            ],
        )

        # ── 처리 시작 ──
        def on_process(
            video_path, mask_mode_str,
            sam_mask, editor_value,
            inpaint_mode_str,
            pod_id, pod_port, pod_url,
            max_res, num_steps, seed, mask_dilation,
            campatch_feather_val,
            campatch_rvm_enabled_val, campatch_rvm_ratio_val,
            campatch_ref_frame_idx,
            pod_connected,
            cfg,
            progress=gr.Progress(),
        ):
            is_campatch = "Simple LaMa" in inpaint_mode_str
            needs_pod = _needs_pod(inpaint_mode_str, campatch_rvm_enabled_val)
            logger.info(
                f"on_process: mask_mode={mask_mode_str}, inpaint={inpaint_mode_str}, "
                f"sam_mask={'yes' if sam_mask is not None else 'no'}, "
                f"needs_pod={needs_pod}, pod_connected={pod_connected}"
            )

            if video_path is None:
                return None, "오류: 영상이 업로드되지 않았습니다"

            if needs_pod and not pod_connected:
                return None, "오류: RunPod 연결이 필요합니다. 상단의 RunPod 연결 설정을 열고 연결하세요."

            # 마스크 결정 — 모드에 따라 엄격하게 분리
            if "SAM" in mask_mode_str:
                if sam_mask is None or sam_mask.max() == 0:
                    return None, "오류: SAM 마스크가 없습니다. 이미지를 클릭하여 대상을 선택하세요."
                mask_raw = sam_mask
            else:
                if editor_value is None:
                    return None, "오류: 마스크가 그려지지 않았습니다"
                layers = editor_value.get("layers", [])
                if not layers:
                    return None, "오류: 브러시로 마스크를 칠해주세요"
                mask_raw = layers[0]
                if not isinstance(mask_raw, np.ndarray):
                    return None, "오류: 마스크 형식을 인식할 수 없습니다"
                if mask_raw.ndim == 3 and mask_raw.shape[2] == 4:
                    has_content = mask_raw[:, :, 3].max() > 0
                else:
                    has_content = mask_raw.max() > 0
                if not has_content:
                    return None, "오류: 브러시로 마스크 영역을 칠해주세요"

            # Pod 연결 설정
            if needs_pod:
                url_clean = (pod_url or "").strip()
                id_clean = (pod_id or "").strip()
                cfg.runpod.custom_url = url_clean
                cfg.runpod.pod_id = id_clean
                cfg.runpod.port = int(pod_port or 8000)

            try:
                video_path = _to_browser_mp4(video_path)
            except Exception as e:
                return None, f"오류: 영상 변환 실패 — {e}"

            cfg.mask.dilation_kernel_size = int(mask_dilation)

            if is_campatch:
                cfg.campatch.feather_radius = int(campatch_feather_val)
                cfg.campatch.rvm_enabled = bool(campatch_rvm_enabled_val)
                cfg.campatch.rvm_downsample_ratio = float(campatch_rvm_ratio_val)
                from .campatch import process_video_campatch

                try:
                    def progress_cb(p: ProcessingProgress):
                        progress(p.percent / 100.0, desc=p.message)

                    output_path = process_video_campatch(
                        video_path, mask_raw, cfg,
                        ref_frame_idx=int(campatch_ref_frame_idx),
                        progress_callback=progress_cb,
                    )
                    return output_path, "CamPatch 처리 완료!"
                except Exception as e:
                    logger.exception("CamPatch 처리 실패")
                    return None, f"오류: {str(e)}"
            else:
                cfg.video.max_inpaint_resolution = int(max_res)
                cfg.minimax_remover.num_inference_steps = int(num_steps)
                cfg.minimax_remover.seed = int(seed)

                agent = CamRemoverAgent(cfg)
                try:
                    def progress_cb(p: ProcessingProgress):
                        progress(p.percent / 100.0, desc=p.message)

                    output_path = agent.process_video(
                        video_path, mask_raw, progress_callback=progress_cb
                    )
                    return output_path, "처리 완료!"
                except Exception as e:
                    logger.exception("처리 실패")
                    return None, f"오류: {str(e)}"

        process_btn.click(
            fn=on_process,
            inputs=[
                state_video_path, mask_mode,
                state_sam_mask, state_brush_mask,
                inpaint_mode,
                pod_id_input, pod_port_input, pod_url_input,
                max_res_input, num_steps_input,
                seed_input, mask_dilation_input,
                campatch_feather,
                campatch_rvm_enabled, campatch_rvm_ratio,
                frame_slider,
                state_pod_connected,
                state_config,
            ],
            outputs=[result_video, progress_text],
        )

        # ── 푸터 ──
        gr.HTML(
            '<div style="text-align:center; margin-top:24px; padding:12px; '
            'border-top:1px solid #e5e7eb; color:#9ca3af; font-size:12px;">'
            'SBS Camremover v1.0.0 &nbsp;|&nbsp; 문의: '
            '<a href="mailto:eight@sbs.co.kr" style="color:#9ca3af;">eight@sbs.co.kr</a>'
            '</div>'
        )

    return app


def launch():
    """Gradio 앱을 시작한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = load_config()
    app = create_ui(config)
    print("\n접속 주소: http://127.0.0.1:7860\n")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    launch()
