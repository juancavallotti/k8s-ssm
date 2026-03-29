# k8s-ssm

A production-ready, GPU-accelerated chatbot system deployed on AWS EKS. The backend runs `cartesia-ai/Llamba-8B` via HuggingFace Transformers; the frontend is a React SPA served by a FastAPI proxy. Everything is containerised, orchestrated with Kubernetes, and provisioned with Terraform.

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
| `GET /api/health` | llm-service | Returns model name + device |

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
│       ├── main.py           FastAPI + HuggingFace Transformers
│       ├── requirements.txt
│       └── Dockerfile        CUDA 12.1 base; model baked in at build time
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

The LLM model is gated. Before building any image:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Request access to [`cartesia-ai/Llamba-8B`](https://huggingface.co/cartesia-ai/Llamba-8B)
3. Generate a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

> **Important**: `HF_TOKEN` is only needed at **image build time**. The tokenizer and model weights are baked into the Docker image during the build step. Running containers do not require the token.

---

## Local Development

### Option A — Dev mode (no GPU required, 1B model)

This is the fastest way to get a working end-to-end stack on any machine. The `ENV=dev` flag swaps the 8B model for the much smaller 1B variant.

```bash
# 1. Clone and set up env
git clone https://github.com/<your-org>/k8s-ssm.git
cd k8s-ssm
cp .env.example .env

# 2. Edit .env — set your HF_TOKEN and enable dev mode
#    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
#    ENV=dev

# 3. Build and start the full stack
docker compose --profile llm up --build

# 4. Open the chat UI
open http://localhost:3000
```

Health checks:
```bash
curl http://localhost:8001/api/health   # LLM service — shows model name and device
curl http://localhost:3000/health       # Chatbot backend
```

### Option B — Full GPU stack (production model locally)

Requires an NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

```bash
cp .env.example .env
# Edit .env: set HF_TOKEN, leave ENV unset (or remove the ENV= line)

docker compose --profile llm up --build
# The LLM container will log "[startup] Model loaded." after ~5 minutes on first run.

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

# LLM image (HF_TOKEN required at build time)
source .env    # loads HF_TOKEN into shell
docker build --build-arg HF_TOKEN=$HF_TOKEN \
             -t llm-service:latest services/llm/
docker tag  llm-service:latest $ECR/llm-service:latest
docker push $ECR/llm-service:latest
```

> **Note**: The LLM image will be large (~20 GB) because the model weights are baked in. The initial push will take several minutes depending on your connection.

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
# A tokenizer download error means HF_TOKEN was missing at build time.
# Rebuild the image with --build-arg HF_TOKEN=<token> and push again.
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

- **Terraform state** is local by default. For team use, add an S3 backend + DynamoDB lock table to `infra/versions.tf`.
- **`package-lock.json`**: The chatbot Dockerfile runs `npm install --legacy-peer-deps`. After the first successful install, commit the generated `package-lock.json` and switch the Dockerfile line to `npm ci --legacy-peer-deps` for reproducible builds.
- **Static IPs**: AWS ALB does not support Elastic IPs directly. The `aws_eip` resources in Terraform are pre-allocated for a potential NLB in front of the ALB. For a truly static public IP, use [AWS Global Accelerator](https://aws.amazon.com/global-accelerator/).
- **GPU taint**: GPU nodes carry the taint `nvidia.com/gpu=true:NoSchedule`. Only the LLM deployment has the matching toleration — no other workloads will be scheduled there accidentally.
