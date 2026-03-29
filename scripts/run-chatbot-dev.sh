#!/usr/bin/env bash
# Run the chatbot FastAPI backend natively for local development.
# Points at the native LLM service (default: http://localhost:8001).
#
# Started automatically by: npm run dev (via concurrently)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/services/chatbot/backend"
VENV_DIR="$BACKEND_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "→ Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --quiet -r "$BACKEND_DIR/requirements.txt"

export LLM_SERVICE_URL="${LLM_SERVICE_URL:-http://localhost:8001}"
echo "→ Starting chatbot backend on http://localhost:8000 (LLM_SERVICE_URL=$LLM_SERVICE_URL)"

cd "$BACKEND_DIR"
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
