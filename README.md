# k8s-ssm

A production-ready, GPU-accelerated chatbot system deployed on AWS EKS. The backend runs `cartesia-ai/Llamba-8B` via the [cartesia-pytorch](https://github.com/cartesia-ai/edge/tree/main/cartesia-pytorch) library (a Mamba-2 SSM, not a Transformer); the frontend is a React SPA served by a FastAPI proxy. Everything is containerised, orchestrated with Kubernetes, and provisioned with Terraform.

---

## Architecture

```
                          ┌─────────────────────────────────────────┐
                          │              AWS EKS Cluster             │
Internet                  │                                          │
   │                      │   ┌─────────────────┐                   │
   │   /          ──────► │   │ chatbot (FastAPI │                   │
   │                      │   │  + React SPA)   │──► POST /api/generate
   │   /api/*     ──────► │   └─────────────────┘         │         │
   │                      │                         ┌──────┴──────┐  │
   └── ALB Ingress ──────►│                         │ llm-service │  │
                          │                         │ (Llamba-8B) │  │
                          │                         │  GPU node   │  │
                          │                         └─────────────┘  │
                          └─────────────────────────────────────────┘
```

| Path | Service | Notes |
|---|---|---|
| `GET /` and all SPA routes | chatbot | Serves React bundle |
| `POST /chat` | chatbot | Proxies to LLM service |
| `GET /health` | chatbot | Liveness / readiness |
| `POST /api/generate` | llm-service | Raw LLM inference |
| `GET /api/health` | llm-service | Returns model name, tokenizer name, and device |

---

## Repository Structure

```
k8s-ssm/
├── services/
│   ├── chatbot/
│   │   ├── backend/          FastAPI proxy + SPA server
│   │   │   ├── main.py
│   │   │   ├── requirements.txt
│   │   │   └── .env.example
│   │   ├── frontend/         React SPA
│   │   │   ├── src/
│   │   │   └── public/
│   │   └── Dockerfile        Multi-stage: Node build → Python serve
│   └── llm/
│       ├── main.py           FastAPI + cartesia-pytorch (LlambaLMHeadModel)
│       ├── requirements.txt
│       └── Dockerfile        CUDA 12.1 devel base; compiles cartesia-pytorch from source
├── k8s/                      Kubernetes manifests (Kustomize)
├── infra/                    Terraform — VPC, EKS, ALB controller
├── plans/                    Implementation plans and status tracking
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Prerequisites

### For local development

| Tool | Minimum version | Notes |
|---|---|---|
| Docker | 24+ | Required |
| Docker Compose | v2 (plugin) | `docker compose` not `docker-compose` |
| Node.js | 20+ | Only for frontend hot-reload dev mode |
| Python | 3.11+ | Only for backend hot-reload dev mode |
| NVIDIA Container Toolkit | latest | Only needed for full GPU stack locally |

### For AWS deployment

| Tool | Minimum version |
|---|---|
| AWS CLI | v2 |
| Terraform | >= 1.9.0 |
| kubectl | >= 1.29 |
| Docker | 24+ |

### HuggingFace access

Three gated repositories must be accessible with your token before building any image:

| Repository | Purpose |
|---|---|
| [`cartesia-ai/Llamba-8B`](https://huggingface.co/cartesia-ai/Llamba-8B) | Production model weights |
| [`cartesia-ai/Llamba-1B`](https://huggingface.co/cartesia-ai/Llamba-1B) | Dev / smoke-test model weights |
| [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B) | Tokenizer for Llamba-8B |
| [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) | Tokenizer for Llamba-1B |

1. Create an account at [huggingface.co](https://huggingface.co)
2. Request access to each of the four repositories above (the Meta LLaMA repos require accepting Meta's license)
3. Generate a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

> **Important**: `HF_TOKEN` is only needed at **image build time**. The model weights and tokenizer files are baked into the Docker image during build. Running containers do not require the token.

---

## Local Development

### Option A — Dev mode (1B model, CPU inference)

The `ENV=dev` flag loads `Llamba-1B` with its lighter tokenizer (`meta-llama/Llama-3.2-1B`). Inference runs on CPU when no CUDA GPU is present, which is slower but functional for smoke-testing.

> **Apple Silicon (M1/M2/M3)**: `cartesia-pytorch` compiles CUDA C++ extensions and does not support `linux/arm64`. You must pass `--platform linux/amd64` so Docker builds an x86_64 image under emulation. The build will take longer (~30–60 min) but the resulting container runs on CPU.

```bash
# 1. Clone and set up env
git clone https://github.com/<your-org>/k8s-ssm.git
cd k8s-ssm
cp .env.example .env

# 2. Edit .env — set your HF_TOKEN and enable dev mode
#    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
#    ENV=dev

# 3. Build and start (add --platform linux/amd64 on Apple Silicon)
docker compose --profile llm up --build
# Apple Silicon:
# DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose --profile llm up --build

# 4. Open the chat UI
open http://localhost:3000
```

Health checks:
```bash
curl http://localhost:8001/api/health   # shows model name, tokenizer, and device
curl http://localhost:3000/health       # chatbot backend
```

### Option B — Full GPU stack (production model locally)

Requires an x86_64 Linux machine with an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

```bash
cp .env.example .env
# Edit .env: set HF_TOKEN, leave ENV unset (or remove the ENV= line)

docker compose --profile llm up --build
# cartesia-pytorch compiles CUDA extensions during the build — expect 20–40 min on first run.
# The LLM container will log "[startup] Model loaded." ~5 minutes after the container starts.

open http://localhost:3000
```

### Option C — Chatbot only (stub the LLM)

Run just the chatbot frontend/backend without any LLM service. Chat requests will return a `503` error from the UI — useful for testing the UI or backend routing in isolation.

```bash
docker compose up --build
open http://localhost:3000
```

### Option D — Hot-reload development

Best for iterating quickly on the backend or frontend without rebuilding Docker images.

```bash
# Terminal 1 — LLM service (optional; use dev mode to avoid GPU requirement)
ENV=dev docker compose --profile llm up llm

# Terminal 2 — FastAPI backend with auto-reload
cd services/chatbot/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
LLM_SERVICE_URL=http://localhost:8001 uvicorn main:app --reload --port 8000

# Terminal 3 — React dev server (proxies /chat → localhost:8000 via package.json proxy)
cd services/chatbot/frontend
npm install
npm start
# Opens http://localhost:3000 with hot reload
```

---

## AWS Deployment

### Step 1 — Configure AWS credentials

```bash
aws configure
# Enter: AWS Access Key ID, Secret Access Key, region (us-east-1), output format (json)

# Verify
aws sts get-caller-identity
```

### Step 2 — Provision infrastructure with Terraform

This creates the VPC, EKS cluster (app nodes + GPU nodes), IAM roles, and the AWS Load Balancer Controller via Helm.

```bash
cd infra
terraform init
terraform plan -var-file=example.tfvars    # Review ~50-70 resources
terraform apply -var-file=example.tfvars   # Takes 15–25 minutes
```

Default configuration (`example.tfvars`):

| Resource | Type | Count |
|---|---|---|
| Region | us-east-1 | — |
| App nodes | t3.medium | 2 (min 1, max 4) |
| GPU nodes | g4dn.xlarge | 1 (min 0, max 2) |
| VPC | 10.0.0.0/16 | 3 AZs |

### Step 3 — Configure kubectl

```bash
aws eks update-kubeconfig --region us-east-1 --name k8s-ssm

# Verify
kubectl get nodes
# You should see 3 nodes: 2 app-nodes + 1 gpu-node
```

### Step 4 — Create ECR repositories

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

aws ecr create-repository --repository-name chatbot      --region $REGION
aws ecr create-repository --repository-name llm-service  --region $REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
```

### Step 5 — Build and push images

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
ECR=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# Chatbot image (no token needed)
docker build -t chatbot:latest services/chatbot/
docker tag  chatbot:latest $ECR/chatbot:latest
docker push $ECR/chatbot:latest

# LLM image — must be built for linux/amd64 (CUDA, required by cartesia-pytorch)
# HF_TOKEN is needed to download Llamba-8B and the meta-llama/Llama-3.1-8B tokenizer.
source .env    # loads HF_TOKEN into shell
docker build --platform linux/amd64 \
             --build-arg HF_TOKEN=$HF_TOKEN \
             -t llm-service:latest services/llm/
docker tag  llm-service:latest $ECR/llm-service:latest
docker push $ECR/llm-service:latest
```

> **Note**: The LLM image is large (~25 GB) — the `devel` CUDA base image, compiled cartesia-pytorch extensions, and baked-in model weights all contribute. The initial build takes 20–40 min and the push several more minutes.

### Step 6 — Update Kubernetes manifests with ECR image URIs

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
ECR=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com

sed -i "s|<YOUR_ECR_REPO>/chatbot:latest|$ECR/chatbot:latest|g"       k8s/chatbot-deployment.yaml
sed -i "s|<YOUR_ECR_REPO>/llm:latest|$ECR/llm-service:latest|g"       k8s/llm-deployment.yaml
```

### Step 7 — Deploy to EKS

```bash
# Apply namespace first (other manifests depend on it)
kubectl apply -f k8s/namespace.yaml

# Apply everything else
kubectl apply -f k8s/

# Watch rollout
kubectl rollout status deployment/chatbot     -n chatbot
kubectl rollout status deployment/llm-service -n chatbot
# Note: the LLM pod has a 300s initial delay — it is loading model weights.
```

### Step 8 — Get the public URL

```bash
# Wait ~2–3 minutes for the ALB to provision, then:
kubectl get ingress -n chatbot

# The ADDRESS column shows the ALB DNS name, e.g.:
# k8s-chatbot-chatbotin-xxxx.us-east-1.elb.amazonaws.com
```

Open that URL in your browser — the chat UI should load immediately.

> **Custom domain**: Point a `CNAME` record at the ALB DNS name in Route 53 (or your DNS provider) for a cleaner public URL.

---

## Environment Variables

| Variable | Service | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | llm (build-time only) | — | HuggingFace API token. Required at `docker build` time; not needed at runtime. |
| `ENV` | llm | `""` | Set to `dev` to load `Llamba-1B` instead of `Llamba-8B` for CPU-friendly smoke-testing. |
| `LLM_SERVICE_URL` | chatbot | `http://localhost:8001` | Base URL of the LLM service. In K8s this is set via ConfigMap to `http://llm-service:8000`. |

---

## Cost Estimate (us-east-1, 24/7)

| Resource | Type | Est. monthly |
|---|---|---|
| EKS Control Plane | Managed | ~$73 |
| app-nodes × 2 | t3.medium | ~$60 |
| gpu-node × 1 | g4dn.xlarge | ~$380 |
| NAT Gateway | Single AZ | ~$32 |
| ALB | Per hour + LCU | ~$20 |
| ECR storage | ~25 GB | ~$2.50 |
| **Total** | | **~$570/month** |

**Cost-saving tip**: Set `desired_size = 0` on the `gpu-nodes` group in Terraform and re-apply when the GPU node is not needed. This saves ~$380/month during inactive periods.

---

## Teardown

> Always delete the Kubernetes namespace **before** running `terraform destroy`. The ALB is managed by the in-cluster Load Balancer Controller; destroying the cluster first may leave the ALB orphaned and still incurring charges.

```bash
# 1. Remove Kubernetes workloads (this also releases the ALB)
kubectl delete namespace chatbot

# 2. Destroy all AWS infrastructure
cd infra
terraform destroy -var-file=example.tfvars

# 3. (Optional) Delete ECR repositories and their images
REGION=us-east-1
aws ecr delete-repository --repository-name chatbot     --force --region $REGION
aws ecr delete-repository --repository-name llm-service --force --region $REGION
```

---

## Troubleshooting

**LLM pod stuck in `Pending`**
```bash
kubectl describe pod -l app=llm-service -n chatbot
# Look for: "0/1 nodes are available: 1 Insufficient nvidia.com/gpu"
# Ensure the NVIDIA device plugin DaemonSet is running on the GPU node:
kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system
```

**LLM pod in `CrashLoopBackOff` on startup**
```bash
kubectl logs -l app=llm-service -n chatbot --previous
# "cannot import cartesia_pytorch" → the image was built without cartesia-pytorch.
#   Rebuild with --platform linux/amd64 and push again.
# Tokenizer/model download error → HF_TOKEN was missing or lacked access to one of:
#   cartesia-ai/Llamba-8B, meta-llama/Llama-3.1-8B
#   Rebuild with --build-arg HF_TOKEN=<token> after granting HuggingFace access.
```

**Chat returns `502 Bad Gateway`**
```bash
# The chatbot reached the LLM service but it returned an error.
kubectl logs -l app=chatbot     -n chatbot
kubectl logs -l app=llm-service -n chatbot
```

**ALB not provisioning (`ADDRESS` column empty after 5 min)**
```bash
kubectl describe ingress chatbot-ingress -n chatbot
# Check the AWS Load Balancer Controller is running:
kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller
```

**Terraform `Error: configuring Terraform AWS Provider`**
```bash
# Verify credentials and region are set correctly.
aws sts get-caller-identity
```

---

## Notes

- **cartesia-pytorch on ARM**: The library compiles CUDA C++ extensions (mamba-ssm, flash-attn, causal-conv1d) and does not support `linux/arm64`. Always build the LLM image with `--platform linux/amd64`. On Apple Silicon this uses QEMU emulation and is significantly slower.
- **LLM image size**: The `devel` CUDA base image (~5 GB), compiled extensions, and baked-in model weights (~15 GB for 8B) produce an image of ~25 GB. Use ECR lifecycle policies to limit stored versions.
- **Terraform state** is local by default. For team use, add an S3 backend + DynamoDB lock table to `infra/versions.tf`.
- **`package-lock.json`**: The chatbot Dockerfile runs `npm install --legacy-peer-deps`. After the first successful install, commit the generated `package-lock.json` and switch the Dockerfile line to `npm ci --legacy-peer-deps` for reproducible builds.
- **Static IPs**: AWS ALB does not support Elastic IPs directly. The `aws_eip` resources in Terraform are pre-allocated for a potential NLB in front of the ALB. For a truly static public IP, use [AWS Global Accelerator](https://aws.amazon.com/global-accelerator/).
- **GPU taint**: GPU nodes carry the taint `nvidia.com/gpu=true:NoSchedule`. Only the LLM deployment has the matching toleration — no other workloads will be scheduled there accidentally.
