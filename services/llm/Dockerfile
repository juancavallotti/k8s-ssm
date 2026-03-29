FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app
COPY requirements.txt .
# On linux/amd64 (production) install CUDA-enabled torch from the PyTorch wheel index.
# On linux/arm64 (Apple Silicon local dev) fall back to the PyPI CPU build — CUDA is
# unavailable on ARM and the cu121 index only carries torch 2.0.x for that platform.
ARG TARGETPLATFORM
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
      pip install --no-cache-dir torch==2.5.1; \
    else \
      pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121; \
    fi \
    && pip install --no-cache-dir -r requirements.txt

# HF_TOKEN is required at BUILD TIME because the LLaMA tokenizer files are
# gated and cannot be downloaded without authentication.
# Pass it with: docker build --build-arg HF_TOKEN=hf_xxx ...
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# ENV controls which model is baked into the image:
#   ENV=dev  → cartesia-ai/Llamba-1B  (fast local smoke-testing, no GPU needed)
#   ENV=<anything else> → cartesia-ai/Llamba-8B  (production)
ARG ENV
ENV ENV=${ENV}

# Pre-download the appropriate model weights during build so the container
# starts instantly at runtime without needing network access or the token.
RUN python3 -c "\
import os; \
from transformers import AutoTokenizer, AutoModelForCausalLM; \
model = 'cartesia-ai/Llamba-1B' if os.getenv('ENV') == 'dev' else 'cartesia-ai/Llamba-8B'; \
print(f'[build] pre-downloading {model}'); \
AutoTokenizer.from_pretrained(model); \
AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto') \
"

# Clear the token from the environment after download (do not leak into runtime)
ENV HF_TOKEN=""

COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
