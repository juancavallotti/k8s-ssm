# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

### Added
- `.github/workflows/build-llm.yml`: builds LLM Docker image on native linux/amd64 (no QEMU), pushes to ECR, and rollout-restarts the EKS pod; fixes CUDA extension segfaults caused by QEMU-compiled kernels on Apple Silicon
- `.github/workflows/build-chatbot.yml`: same pattern for the chatbot image
- Per-service Makefile targets: `build-llm`, `build-chatbot`, `push-llm`, `push-chatbot`

### Added
- `Makefile` at repo root: sequences `infra-base` → `infra-helm` → `build` → `push` → `setup-k8s` → `deploy` to avoid terraform chicken-and-egg; auto-creates ECR repos; validates `HF_TOKEN` before creating k8s secret; handles EKS access entry for current IAM identity
- `k8s/chatbot-deployment.yaml.template` and `k8s/llm-deployment.yaml.template`: deployment manifests with `{{ECR_BASE}}` placeholder; rendered files are gitignored so account IDs are never committed

### Changed
- `infra/main.tf`: replaced `disk_size` with `block_device_mappings` (gp3, 50GB app / 150GB GPU) — `disk_size` is silently ignored by EKS module v20+ when using managed launch templates
- `services/llm/main.py`: load pytorch model with `device_map` and `torch_dtype` to avoid staging 16GB weights in CPU RAM before GPU transfer (OOMKill fix)

### Added
- Native local dev workflow: `Procfile` + `honcho` (`pip install honcho`, then `honcho start`); starts LLM service, chatbot backend, and frontend in one command
- `scripts/run-llm-dev.sh`: auto-detects Apple Silicon (MLX via `cartesia-mlx`) vs Intel (CPU pytorch); handles venv setup
- `scripts/run-chatbot-dev.sh`: venv setup and hot-reload for chatbot FastAPI backend
- LLM `main.py` now supports two backends: MLX (Apple Silicon, auto-detected) and PyTorch (CUDA/CPU fallback); MLX uses `cartesia-ai/Llamba-*-4bit-mlx` model variants

### Changed
- Frontend migrated from Create React App (`react-scripts`) to Vite + TypeScript; added Tailwind CSS v4, Lucide React icons, and `react-markdown` with GFM; chat UI rebuilt with Tailwind utility classes and markdown rendering for assistant responses
- Chatbot Dockerfile updated: Node 18 → Node 20, build output path `build/` → `dist/`, dropped `--legacy-peer-deps`
- LLM Dockerfile: install `cartesia-pytorch` from GitHub source instead of PyPI (`0.0.2` sdist is missing `version.py`)
- `docker-compose.yml`: `LLM_SERVICE_URL` is now configurable via env var (default `http://llm:8000`)

### Added
- Monorepo scaffold: initial directory structure, .gitignore, and README (Stage 1)
- LLM service — FastAPI + `cartesia-ai/Llamba-8B`; `ENV=dev` loads `Llamba-1B` for CPU smoke-testing (Stage 2)
- Chatbot backend — FastAPI proxy with `/chat` and `/health` endpoints, serves React SPA static files (Stage 3)
- Chatbot frontend — React SPA with chat UI; multi-stage Dockerfile (Node build + Python serve) (Stage 4)
- Kubernetes manifests — namespace, configmap, LLM + chatbot deployments/services, ALB ingress, NVIDIA device plugin DaemonSet, kustomization (Stage 5)
- Terraform infrastructure — VPC, EKS with app/GPU node groups, IRSA, ALB controller Helm release, Elastic IPs; ALB IAM policy downloaded from upstream (Stage 6)
- Local dev setup: docker-compose.yml with dev/GPU/chatbot-only profiles, .env.example, full README with local dev options (A–D) and step-by-step AWS deployment guide (Stage 7)
