# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

### Changed
- Frontend migrated from Create React App (`react-scripts`) to Vite + TypeScript; added Tailwind CSS v4, Lucide React icons, and `react-markdown` with GFM; chat UI rebuilt with Tailwind utility classes and markdown rendering for assistant responses
- Chatbot Dockerfile updated: Node 18 → Node 20, build output path `build/` → `dist/`, dropped `--legacy-peer-deps`
- LLM Dockerfile: install `cartesia-pytorch` from GitHub source instead of PyPI (`0.0.2` sdist is missing `version.py`)

### Added
- Monorepo scaffold: initial directory structure, .gitignore, and README (Stage 1)
- LLM service — FastAPI + `cartesia-ai/Llamba-8B`; `ENV=dev` loads `Llamba-1B` for CPU smoke-testing (Stage 2)
- Chatbot backend — FastAPI proxy with `/chat` and `/health` endpoints, serves React SPA static files (Stage 3)
- Chatbot frontend — React SPA with chat UI; multi-stage Dockerfile (Node build + Python serve) (Stage 4)
- Kubernetes manifests — namespace, configmap, LLM + chatbot deployments/services, ALB ingress, NVIDIA device plugin DaemonSet, kustomization (Stage 5)
- Terraform infrastructure — VPC, EKS with app/GPU node groups, IRSA, ALB controller Helm release, Elastic IPs; ALB IAM policy downloaded from upstream (Stage 6)
- Local dev setup: docker-compose.yml with dev/GPU/chatbot-only profiles, .env.example, full README with local dev options (A–D) and step-by-step AWS deployment guide (Stage 7)
