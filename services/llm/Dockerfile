# NOTE: cartesia-pytorch compiles CUDA C++ extensions (mamba-ssm, flash-attn,
# causal-conv1d) and therefore requires:
#   1. The DEVEL image (provides nvcc / CUDA headers).
#   2. linux/amd64 — ARM builds are not supported by the library.
# On Apple Silicon use: docker build --platform linux/amd64 ...
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app
COPY requirements.txt .

# Install torch 2.4.0 with CUDA 12.1 support (required by cartesia-pytorch).
RUN pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install cartesia-pytorch from source (--no-binary compiles CUDA extensions).
RUN pip install --no-cache-dir --no-binary :all: cartesia-pytorch

# Install remaining FastAPI / Uvicorn dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# HF_TOKEN is required at BUILD TIME to download the gated Llamba model weights
# and the corresponding LLaMA tokenizer from HuggingFace.
# Pass it with: docker build --build-arg HF_TOKEN=hf_xxx ...
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# ENV controls which model is baked into the image:
#   ENV=dev  → cartesia-ai/Llamba-1B  + meta-llama/Llama-3.2-1B tokenizer
#   ENV=<anything else> → cartesia-ai/Llamba-8B + meta-llama/Llama-3.1-8B tokenizer
ARG ENV
ENV ENV=${ENV}

# Pre-download model weights and tokenizer during build so the container
# starts instantly at runtime without needing network access or the token.
RUN python3 -c "\
import os; \
from transformers import AutoTokenizer; \
from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel; \
is_dev = os.getenv('ENV') == 'dev'; \
model_name = 'cartesia-ai/Llamba-1B' if is_dev else 'cartesia-ai/Llamba-8B'; \
tok_name   = 'meta-llama/Llama-3.2-1B' if is_dev else 'meta-llama/Llama-3.1-8B'; \
print(f'[build] pre-downloading {model_name} + tokenizer {tok_name}'); \
AutoTokenizer.from_pretrained(tok_name); \
LlambaLMHeadModel.from_pretrained(model_name); \
print('[build] done.') \
"

# Clear the token from the environment after download (do not leak into runtime)
ENV HF_TOKEN=""

COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
