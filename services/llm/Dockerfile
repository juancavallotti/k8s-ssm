# NOTE: cartesia-pytorch compiles CUDA C++ extensions (mamba-ssm, flash-attn,
# causal-conv1d) and therefore requires:
#   1. The DEVEL image (provides nvcc / CUDA headers).
#   2. linux/amd64 — ARM builds are not supported by the library.
# On Apple Silicon use: docker build --platform linux/amd64 ...
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Pin python3/python to 3.11, then bootstrap pip via get-pip.py.
# ensurepip is disabled in Debian/Ubuntu for system Python; get-pip.py bypasses that.
# All packages are installed with `python3 -m pip` to ensure they land in 3.11's
# site-packages (apt's python3-pip targets 3.10 on Ubuntu 22.04).
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && curl -sSL https://bootstrap.pypa.io/get-pip.py | python3

WORKDIR /app
COPY requirements.txt .

# Install torch 2.4.0 with CUDA 12.1 support (required by cartesia-pytorch).
RUN python3 -m pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install pure-Python build/runtime deps that don't need CUDA.
RUN python3 -m pip install --no-cache-dir \
    packaging setuptools wheel einops \
    "transformers==4.39.3"

# causal-conv1d, mamba-ssm, and cartesia-pytorch all import torch inside their
# setup.py, so pip's build isolation (which creates a fresh env without torch)
# will always fail. --no-build-isolation runs setup.py in the current environment
# where torch is already present.
RUN python3 -m pip install --no-cache-dir --no-build-isolation causal-conv1d==1.4.0
RUN python3 -m pip install --no-cache-dir --no-build-isolation mamba-ssm==2.2.2
RUN python3 -m pip install --no-cache-dir psutil
RUN python3 -m pip install --no-cache-dir --no-build-isolation flash-attn==2.6.3

# cartesia_pytorch is pure Python — the CUDA kernels are in causal-conv1d / mamba-ssm /
# flash-attn (already compiled above). setup.py tries to compile CUDA extensions and
# exits non-cleanly under emulation, leaving the Python files uninstalled even though
# pip reports success. Copying the source directly is simpler and fully reliable.
RUN git clone --depth=1 https://github.com/cartesia-ai/edge.git /tmp/cartesia-edge \
    && cp -r /tmp/cartesia-edge/cartesia-pytorch/cartesia_pytorch \
       /usr/local/lib/python3.11/dist-packages/cartesia_pytorch \
    && rm -rf /tmp/cartesia-edge

# Install remaining FastAPI / Uvicorn dependencies.
RUN python3 -m pip install --no-cache-dir -r requirements.txt

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
# snapshot_download fetches files to the HF cache without loading tensors
# into RAM — safe under QEMU/memory-constrained build environments.
RUN python3 -c "\
import os; \
from transformers import AutoTokenizer; \
from huggingface_hub import snapshot_download, login; \
token = os.getenv('HF_TOKEN'); \
login(token=token); \
is_dev = os.getenv('ENV') == 'dev'; \
model_name = 'cartesia-ai/Llamba-1B' if is_dev else 'cartesia-ai/Llamba-8B'; \
tok_name   = 'meta-llama/Llama-3.2-1B'; \
print(f'[build] pre-downloading {model_name} + tokenizer {tok_name}'); \
AutoTokenizer.from_pretrained(tok_name, token=token); \
snapshot_download(repo_id=model_name, token=token); \
print('[build] done.') \
"

# Clear the token from the environment after download (do not leak into runtime)
ENV HF_TOKEN=""

COPY main.py .
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
