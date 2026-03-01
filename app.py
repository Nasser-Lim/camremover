"""Vercel / 직접 실행 엔트리포인트."""

import os
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.config import load_config
from agent.ui import create_ui
import gradio as gr

config = load_config()
demo = create_ui(config)

# Vercel ASGI 핸들러: Gradio Blocks를 FastAPI에 마운트
from fastapi import FastAPI
_fastapi = FastAPI()
app = gr.mount_gradio_app(_fastapi, demo, path="/")

if __name__ == "__main__":
    from agent.ui import launch
    launch()
