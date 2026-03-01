FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ProPainter 클론
RUN git clone https://github.com/sczhou/ProPainter.git /workspace/ProPainter

# 모델 가중치 다운로드
RUN mkdir -p /workspace/weights && \
    wget -q -O /workspace/weights/raft-things.pth \
      "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth" && \
    wget -q -O /workspace/weights/recurrent_flow_completion.pth \
      "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth" && \
    wget -q -O /workspace/weights/ProPainter.pth \
      "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth"

# Python 의존성
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# 서버 코드
COPY server.py /workspace/server.py
COPY propainter_inference.py /workspace/propainter_inference.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
