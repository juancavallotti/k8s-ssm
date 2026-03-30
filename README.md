# k8s-ssm

A production-ready, GPU-accelerated chatbot system deployed on AWS EKS. The backend runs `cartesia-ai/Llamba-3B` via the [cartesia-pytorch](https://github.com/cartesia-ai/edge/tree/main/cartesia-pytorch) library (a Mamba-2 SSM, not a Transformer); the frontend is a React SPA served by a FastAPI proxy. Everything is containerised, orchestrated with Kubernetes, and provisioned with Terraform.

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
                          │                         │ (Llamba-3B) │  │
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

## State Space Models — Why Not a Transformer?

Most LLM deployments use Transformer-based models (GPT, LLaMA, Mistral, etc.). This project instead uses **Llamba**, a hybrid SSM (State Space Model) built on [Mamba-2](https://arxiv.org/abs/2405.21060) — a fundamentally different sequence modelling architecture.

### How Transformers work

Transformers process sequences using **self-attention**: every token attends to every other token in the context window. This is powerful but expensive:

- **Time complexity**: O(n²) in sequence length — doubling the context quadruples the compute
- **Memory**: the KV cache grows linearly with context length; long conversations consume significant GPU RAM
- **Parallelism**: attention is computed in one shot over the whole sequence, which suits training but wastes work during inference (you recompute attention for tokens you've already seen)

### How SSMs work

State Space Models represent sequence processing as a **linear dynamical system**:

```
h(t) = A · h(t-1) + B · x(t)   ← update hidden state
y(t) = C · h(t)                 ← produce output
```

Where `h(t)` is a fixed-size hidden state, `x(t)` is the current input token, and `A`, `B`, `C` are learned matrices. During inference this reduces to a simple **recurrence** — processing each new token requires only the current hidden state, not the full history.

**Mamba** (2023) made SSMs competitive with Transformers by making `A`, `B`, `C` input-dependent (selective state space), allowing the model to decide what to remember and what to forget.

**Mamba-2 / Structured State Space Duality (SSD)** (2024) reformulated Mamba as a matrix multiplication that can be computed efficiently in parallel during training (like attention) while still running as a recurrence at inference time.

**Llamba** combines the Llama 3 architecture (RoPE embeddings, RMSNorm, SwiGLU MLP) with Discrete Mamba-2 mixer layers in place of multi-head attention.

### Why this matters for inference

| Property | Transformer | SSM (Mamba-2) |
|---|---|---|
| Per-token inference compute | O(n) — scales with context | O(1) — constant regardless of context |
| KV / state cache size | Grows with context length | Fixed size (hidden state only) |
| Memory bandwidth | High (read entire KV cache per step) | Low (read/write fixed state) |
| Long-context capability | Degrades beyond training window | Handles arbitrarily long contexts |

For a chatbot with long conversations, SSMs maintain constant inference speed and memory usage regardless of how much has been said. A Transformer slows down and uses more memory as the conversation grows.

---

## GPU Scheduling in Kubernetes

Running a GPU workload on Kubernetes requires three things to line up: a node with a physical GPU, a Kubernetes plugin that exposes it as a schedulable resource, and pod configuration that requests it correctly.

### Node group and taint

The Terraform configuration provisions a dedicated **GPU node group** (`g5.xlarge`, 1 NVIDIA A10G, 24 GB VRAM) separate from the application node group (`t3.medium`). The GPU node carries a **taint**:

```
nvidia.com/gpu=true:NoSchedule
```

A taint is a repellent — any pod that doesn't explicitly tolerate it will never be scheduled on that node. This prevents CPU-only workloads (the chatbot, system pods) from accidentally consuming GPU node resources, which are expensive (~$580/month for a g5.xlarge).

### NVIDIA device plugin

Raw Kubernetes has no concept of GPUs. The [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) is a DaemonSet that runs on every GPU node, discovers the physical GPUs via the NVIDIA Container Toolkit, and registers them with the Kubelet as an extended resource: `nvidia.com/gpu`.

Once registered, pods can request GPUs the same way they request CPU and memory:

```yaml
resources:
  requests:
    nvidia.com/gpu: "1"
  limits:
    nvidia.com/gpu: "1"
```

Kubernetes enforces GPU requests as exclusive allocations — two pods cannot share a single GPU (unless MIG partitioning is configured, which it is not here).

### LLM pod configuration

The LLM deployment ([k8s/llm-deployment.yaml.template](k8s/llm-deployment.yaml.template)) brings together all three:

```yaml
spec:
  nodeSelector:
    role: gpu-nodes          # only schedule on the GPU node group
  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"   # tolerate the GPU taint
  containers:
    - resources:
        requests:
          nvidia.com/gpu: "1"  # request one GPU
        limits:
          nvidia.com/gpu: "1"
```

The `nodeSelector` ensures the pod lands on a GPU node. The toleration overrides the taint that would otherwise block it. The resource request tells the scheduler that this pod needs one GPU, and the limit ensures it gets exclusive access.

No other deployment in this repo has the toleration, so only the LLM pod ever runs on the GPU node.

### Deployment strategy

The LLM deployment uses `strategy: Recreate` rather than the default `RollingUpdate`. With `RollingUpdate`, Kubernetes tries to spin up the new pod before terminating the old one — but since there is only one GPU and each pod requests all of it, the new pod can never be scheduled until the old one is gone. `Recreate` terminates the old pod first, freeing the GPU, then starts the new one.

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
| make | any |

### HuggingFace access

Three gated repositories must be accessible with your token at **runtime** (model weights are downloaded to a PersistentVolume on first pod start, not baked into the image):

| Repository | Purpose |
|---|---|
| [`cartesia-ai/Llamba-3B`](https://huggingface.co/cartesia-ai/Llamba-3B) | Production model weights (~6GB VRAM in bfloat16) |
| [`cartesia-ai/Llamba-1B`](https://huggingface.co/cartesia-ai/Llamba-1B) | Dev / smoke-test model weights |
| [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) | Tokenizer (shared by both models) |

1. Create an account at [huggingface.co](https://huggingface.co)
2. Request access to each repository above (the Meta LLaMA repo requires accepting Meta's license)
3. Generate a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

> **Important**: `HF_TOKEN` is stored as a Kubernetes secret and injected at runtime. It is **not** needed at image build time.

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

The `docker-compose.gpu.yml` override adds the NVIDIA device reservation. Merge it with the base file using `-f`:

```bash
cp .env.example .env
# Edit .env: set HF_TOKEN, leave ENV unset (or remove the ENV= line)

docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile llm up --build
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
npm run dev
# Opens http://localhost:3000 with hot reload
```

---

## AWS Deployment

### Step 1 — Configure AWS credentials

Create a dedicated IAM user with `AdministratorAccess` (do not use root):

```bash
# Create user and attach policy
aws iam create-user --user-name terraform
aws iam attach-user-policy \
  --user-name terraform \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
aws iam create-access-key --user-name terraform
# Save the AccessKeyId and SecretAccessKey from the output

# Configure a named profile
aws configure --profile terraform
# Enter the access key, secret, region (us-west-2), output format (json)

export AWS_PROFILE=terraform

# Verify
aws sts get-caller-identity
```

> **GPU quota**: New AWS accounts default to 0 G-instance vCPUs. Request a quota increase for `Running On-Demand G and VT instances` to at least 4 in [Service Quotas](https://console.aws.amazon.com/servicequotas/) before deploying.

### Step 2 — Deploy everything with Make

```bash
export HF_TOKEN=hf_your_token_here
make all
```

That's it. `make all` runs these steps in order:

| Step | What it does |
|---|---|
| `infra-base` | Provisions VPC, EKS cluster, node groups, IAM roles (skips Helm to avoid chicken-and-egg) |
| `infra-helm` | Installs AWS Load Balancer Controller via Helm |
| `build` | Builds `chatbot` and `llm` Docker images for `linux/amd64` |
| `push` | Creates ECR repos if missing, authenticates, and pushes both images |
| `setup-k8s` | Creates the `hf-token` Kubernetes secret |
| `deploy` | Applies all Kubernetes manifests and waits for rollout |

Total time: ~25–35 minutes (cluster provisioning dominates).

Default infrastructure (`infra/terraform.tfvars`):

| Resource | Type | Count |
|---|---|---|
| Region | us-west-2 | — |
| App nodes | t3.medium | 2 (min 1, max 4) |
| GPU nodes | g5.xlarge (24 GB VRAM) | 1 (min 0, max 2) |
| VPC | 10.0.0.0/16 | 3 AZs |

### Step 3 — Get the public URL

```bash
# Wait ~2–3 minutes for the ALB to provision, then:
kubectl get ingress -n chatbot
# The ADDRESS column shows the ALB DNS name
```

Open that URL in your browser — the chat UI loads immediately. The LLM pod downloads model weights (~16 GB) to its PersistentVolume on first start; subsequent restarts are fast since the weights are cached.

> **First-run note**: The LLM pod will show `0/1 Running` for 2–5 minutes while downloading model weights from HuggingFace. Watch progress with `kubectl logs -n chatbot -l app=llm-service -f`.

> **Custom domain**: Point a `CNAME` record at the ALB DNS name in your DNS provider for a cleaner public URL.

### Individual make targets

```bash
make infra          # Provision/update AWS infrastructure only
make build          # Rebuild both Docker images locally (QEMU on Apple Silicon)
make build-llm      # Rebuild LLM image only
make build-chatbot  # Rebuild chatbot image only
make push           # Push both images to ECR
make push-llm       # Push LLM image only
make push-chatbot   # Push chatbot image only
make setup-k8s      # Re-create the HF token secret (after token rotation)
make deploy         # Re-apply k8s manifests
make teardown       # Destroy everything
```

---

## CI/CD — GitHub Actions

The LLM image **must** be built on a native linux/amd64 host. Building it locally on Apple Silicon under QEMU emulation produces broken CUDA extension binaries that segfault at runtime. The GitHub Actions workflows fix this by running on `ubuntu-latest` (native x86_64).

### Workflows

| File | Trigger | What it does |
|---|---|---|
| `.github/workflows/build-llm.yml` | Push to `main` touching `services/llm/**`, or manual | Builds LLM image natively → pushes to ECR → rollout restarts the pod |
| `.github/workflows/build-chatbot.yml` | Push to `main` touching `services/chatbot/**`, or manual | Builds chatbot image → pushes to ECR → rollout restarts the pod |

Both workflows can also be triggered manually from the GitHub Actions UI ("Run workflow" button) with an optional `deploy` toggle.

### Setup — GitHub repository secrets

Go to **Settings → Secrets and variables → Actions** in your GitHub repo and add:

| Secret name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | Access key for the IAM user created in Step 1 |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret key |

Optionally, add these as **repository variables** (not secrets) to override defaults:

| Variable name | Default | Description |
|---|---|---|
| `AWS_REGION` | `us-west-2` | AWS region |
| `CLUSTER_NAME` | `k8s-ssm` | EKS cluster name |

The IAM user needs `AmazonEC2ContainerRegistryPowerUser` (for ECR push) and `AmazonEKSClusterPolicy` / `AmazonEKSWorkerNodePolicy` (for `kubectl rollout`). Using the same `AdministratorAccess` user from Step 1 is simplest.

---

## Environment Variables

| Variable | Service | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | llm | — | HuggingFace API token. Injected at runtime via Kubernetes secret. |
| `ENV` | llm | `""` | Set to `dev` to load `Llamba-1B` for CPU-friendly smoke-testing. |
| `MODEL_NAME` | llm | `cartesia-ai/Llamba-3B` (prod) / `cartesia-ai/Llamba-1B` (dev) | Override the model loaded at startup. |
| `TOKENIZER_NAME` | llm | `meta-llama/Llama-3.2-1B` | Override the tokenizer. |
| `HF_CACHE_DIR` | llm | `/model-cache` | Path where model weights are cached. Mapped to the PVC in K8s. |
| `LLM_SERVICE_URL` | chatbot | `http://localhost:8001` | Base URL of the LLM service. In K8s set via ConfigMap to `http://llm-service:8000`. |

---

## Cost Estimate (us-east-1, 24/7)

| Resource | Type | Est. monthly |
|---|---|---|
| EKS Control Plane | Managed | ~$73 |
| app-nodes × 2 | t3.medium | ~$60 |
| gpu-node × 1 | g5.xlarge | ~$580 |
| NAT Gateway | Single AZ | ~$32 |
| ALB | Per hour + LCU | ~$20 |
| ECR storage | ~5 GB (no weights in image) | ~$0.50 |
| EBS PVC | 60 GB gp3 (model cache) | ~$5 |
| **Total** | | **~$770/month** |

**Cost-saving tip**: Set `desired_size = 0` on the `gpu-nodes` group in Terraform and re-apply when the GPU node is not needed. This saves ~$580/month during inactive periods. The PVC retains the cached model weights so the next cold start skips the download.

---

## Teardown

> Always delete the Kubernetes workloads **before** running `terraform destroy`. The ALB is managed by the in-cluster Load Balancer Controller; destroying the cluster first may leave the ALB orphaned and still incurring charges.

```bash
make teardown
```

Or manually:
```bash
# 1. Remove Kubernetes workloads (releases the ALB)
kubectl delete -k k8s/

# 2. Destroy all AWS infrastructure
cd infra && terraform destroy -auto-approve

# 3. (Optional) Delete ECR repositories
REGION=us-west-2
aws ecr delete-repository --repository-name chatbot --force --region $REGION
aws ecr delete-repository --repository-name llm     --force --region $REGION
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

**LLM pod in `CrashLoopBackOff` or `CreateContainerConfigError`**
```bash
kubectl describe pod -l app=llm-service -n chatbot | grep -A5 Warning
# "secret hf-token not found" → run: make setup-k8s (requires HF_TOKEN env var)
kubectl logs -l app=llm-service -n chatbot --previous
# "cannot import cartesia_pytorch" → image was built for wrong arch.
#   Rebuild with --platform linux/amd64: make build && make push
# HuggingFace 403/401 → token lacks access to cartesia-ai/Llamba-3B or meta-llama/Llama-3.2-1B
#   Request access on HuggingFace, then re-run: make setup-k8s
```

**LLM pod `OOMKilled`**
```bash
# The g5.xlarge has 16 GB RAM and 24 GB VRAM.
# The model (~16 GB in bfloat16) must load directly to GPU — not staged in CPU RAM.
# Ensure main.py uses device_map=device in LlambaLMHeadModel.from_pretrained().
kubectl logs -l app=llm-service -n chatbot
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
- **LLM image size**: The `devel` CUDA base image (~5 GB) plus compiled extensions produce an image of ~8 GB. Model weights (~16 GB in bfloat16) are **not** baked into the image — they are downloaded at runtime to a 60 GB EBS PersistentVolume (`llm-model-cache`). Subsequent pod restarts skip the download since the volume is retained.
- **Terraform state** is local by default. For team use, add an S3 backend + DynamoDB lock table to `infra/versions.tf`.
- **`package-lock.json`**: The chatbot Dockerfile runs `npm install`. After the first successful build, commit the generated `package-lock.json` and switch the Dockerfile line to `npm ci` for fully reproducible builds.
- **Static IPs**: AWS ALB does not support Elastic IPs directly. The `aws_eip` resources in Terraform are pre-allocated for a potential NLB in front of the ALB. For a truly static public IP, use [AWS Global Accelerator](https://aws.amazon.com/global-accelerator/).
- **GPU taint**: GPU nodes carry the taint `nvidia.com/gpu=true:NoSchedule`. Only the LLM deployment has the matching toleration — no other workloads will be scheduled there accidentally.
