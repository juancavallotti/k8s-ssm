# Implementation Status

Tracks execution progress of [implementation-plan.md](implementation-plan.md).

## Stages

| Stage | Title | Status | Notes |
|---|---|---|---|
| 1 | Monorepo Scaffold | `done` | |
| 2 | LLM Service | `done` | Requires `HF_TOKEN` at build time; `ENV=dev` → 1B model |
| 3 | Chatbot Backend (FastAPI) | `done` | |
| 4 | Chatbot Frontend (React) | `done` | |
| 5 | Kubernetes Manifests | `done` | |
| 6 | Terraform Infrastructure | `done` | Requires AWS credentials |
| 7 | Local Dev Setup & README | `done` | |

**Status values**: `pending` · `in-progress` · `done` · `blocked`

---

## Change Log

| Date | Stage | Event |
|---|---|---|
| 2026-03-29 | — | Plan written and reviewed |
| 2026-03-29 | — | Dockerfile updated: `HF_TOKEN` required at build time (LLaMA tokenizer gating) |
| 2026-03-29 | 1 | Monorepo scaffold: directories, .gitignore, README created |
| 2026-03-29 | 2 | LLM service: requirements.txt, main.py, Dockerfile; ENV=dev uses Llamba-1B |
| 2026-03-29 | 3 | Chatbot backend: requirements.txt, main.py, .env.example |
| 2026-03-29 | 4 | Chatbot frontend: React SPA + multi-stage Dockerfile |
| 2026-03-29 | 5 | Kubernetes manifests: namespace, configmap, deployments, services, ingress, nvidia plugin, kustomization |
| 2026-03-29 | 6 | Terraform: VPC, EKS (app+gpu node groups), IRSA, ALB controller Helm release, Elastic IPs |
| 2026-03-29 | 7 | Local dev setup: docker-compose.yml, .env.example, full README with local + AWS instructions |
| 2026-03-29 | 4 | Frontend: migrated CRA → Vite + TypeScript + Tailwind v4 + Lucide + react-markdown |
| 2026-03-29 | 2 | LLM Dockerfile: cartesia-pytorch installed from git source (PyPI 0.0.2 broken) |
| 2026-03-29 | 7 | Native dev workflow: concurrently-based npm run dev; MLX backend for Apple Silicon; run-llm-dev.sh + run-chatbot-dev.sh |
