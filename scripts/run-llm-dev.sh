#!/usr/bin/env bash
# Run the LLM service natively on macOS (Apple Silicon → MLX, Intel → CPU pytorch).
#
# Usage:
#   ./scripts/run-llm-dev.sh          # uses 1B model (ENV=dev)
#   ENV=prod ./scripts/run-llm-dev.sh # uses 8B model
#
# The service starts on http://localhost:8001.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLM_DIR="$REPO_ROOT/services/llm"
PORT="${LLM_PORT:-8001}"
ENV="${ENV:-dev}"
ARCH="$(uname -m)"

# ── HuggingFace token ────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" && -f "$REPO_ROOT/.env" ]]; then
  export $(grep -E '^HF_TOKEN=' "$REPO_ROOT/.env" | xargs) 2>/dev/null || true
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. The Llamba models are gated on HuggingFace."
  echo "       Set it in .env or export HF_TOKEN=hf_... before running this script."
  exit 1
fi

# ── Python selection ─────────────────────────────────────────────────────────
# On Apple Silicon, cartesia_mlx requires Metal and must be installed in the
# active Python environment beforehand (e.g. your conda/brew env).
# We find whichever python3 already has cartesia_mlx rather than creating a
# fresh venv that would lack it.

if [[ "$ARCH" == "arm64" ]]; then
  # Find a Python that has cartesia_mlx installed
  PYTHON=""
  for candidate in python3 python3.11 python3.12 python3.13; do
    if command -v "$candidate" &>/dev/null; then
      if "$candidate" -c "import cartesia_mlx" 2>/dev/null; then
        PYTHON="$(command -v "$candidate")"
        break
      fi
    fi
  done

  if [[ -z "$PYTHON" ]]; then
    echo "ERROR: cartesia_mlx not found in any Python on PATH."
    echo "       Install it in your active environment first:"
    echo "         pip install cartesia-metal cartesia-mlx"
    exit 1
  fi

  echo "→ Using $PYTHON (has cartesia_mlx)"
  # Ensure fastapi/uvicorn are available in the same environment
  "$PYTHON" -m pip install --quiet "fastapi==0.115.6" "uvicorn[standard]==0.32.1"

else
  # Intel Mac — CPU-only pytorch via a local venv
  VENV_DIR="$LLM_DIR/.venv"
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "→ Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  source "$VENV_DIR/bin/activate"
  PYTHON="$(command -v python3)"

  pip install --quiet --upgrade pip
  pip install --quiet torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
  pip install --quiet packaging setuptools wheel einops "transformers==4.39.3"
  pip install --quiet --no-build-isolation causal-conv1d==1.4.0
  pip install --quiet --no-build-isolation mamba-ssm==2.2.2
  pip install --quiet psutil
  pip install --quiet --no-build-isolation flash-attn==2.6.3 || \
    echo "⚠  flash-attn skipped (no CUDA) — inference will still work on CPU"

  if ! python3 -c "import cartesia_pytorch" 2>/dev/null; then
    TMP=$(mktemp -d)
    git clone --depth=1 https://github.com/cartesia-ai/edge.git "$TMP/cartesia-edge"
    PY_VER="$("$PYTHON" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
    cp -r "$TMP/cartesia-edge/cartesia-pytorch/cartesia_pytorch" \
      "$VENV_DIR/lib/$PY_VER/site-packages/cartesia_pytorch"
    rm -rf "$TMP"
  fi

  pip install --quiet "fastapi==0.115.6" "uvicorn[standard]==0.32.1"
fi

# ── Launch ───────────────────────────────────────────────────────────────────
echo ""
echo "→ Starting LLM service on http://localhost:$PORT (ENV=$ENV, arch=$ARCH)"
echo ""

export ENV="$ENV"
cd "$LLM_DIR"
exec "$PYTHON" -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
