# Context

This plan builds a Python monorepo (`k8s-ssm`) that deploys a GPU-accelerated chatbot system to AWS EKS. The system has two services: a chatbot (FastAPI + React frontend) that proxies user messages to an LLM service (FastAPI serving `cartesia-ai/Llamba-8B` via HuggingFace Transformers). Infrastructure is provisioned with Terraform (EKS with GPU node group) and routed via a single Ingress (`/` → chatbot, `/api` → LLM).

The repo is currently empty (only `.gitignore` and empty `README.md`). All code must be created from scratch across 7 independent stages.

---

# Multi-Stage Implementation Plan: k8s-ssm Python Monorepo

> Each stage is independently executable by a separate LLM agent. Agents should read this plan in full before starting their assigned stage.

## Dependency Map

```
Stage 1 (Scaffold)
  ├── Stage 2 (LLM Service)
  ├── Stage 3 (Chatbot Backend)   ← depends on Stage 2 API contract
  │       └── Stage 4 (Frontend)  ← depends on Stage 3 API contract
  │               └── Stage 5 (K8s Manifests) ← after Stages 2+3+4
  └── Stage 6 (Terraform Infra)   ← independent of services
Stage 7 (Local Dev + README)      ← after all service stages
```

**Stages 2 and 6 can run in parallel after Stage 1.**

---

## Stage 1: Monorepo Scaffold

**Prerequisites**: None.

### Files to Create

**1. `README.md`** — Replace the empty file with:
```markdown
# k8s-ssm
A Kubernetes-deployed chatbot system with a GPU-accelerated LLM backend.
See Stage 7 for full README content.
```

**2. Append to `.gitignore`**:
```
# Node / React
node_modules/
services/chatbot/frontend/build/
services/chatbot/frontend/.env.local

# Terraform
infra/.terraform/
infra/.terraform.lock.hcl
infra/terraform.tfstate
infra/terraform.tfstate.backup
infra/*.tfvars
!infra/example.tfvars

# Docker
.docker/

# Environment files
.env
*.env
!*.env.example
```

**3. Create directory structure** by placing `.gitkeep` files in:
- `services/chatbot/backend/.gitkeep`
- `services/chatbot/frontend/.gitkeep`
- `services/llm/.gitkeep`
- `infra/.gitkeep`
- `k8s/.gitkeep`
- `plans/.gitkeep`

### Verification
- `ls -R` shows all directories.
- `git status` shows new untracked directories.

---

## Stage 2: LLM Service

**Prerequisites**: Stage 1 complete. `services/llm/` directory exists.

**API contract** (referenced by Stage 3):
- `POST /api/generate` — body: `{"prompt": "...", "max_tokens": 512}` → response: `{"response": "..."}`
- `GET /api/health` → `{"status": "ok", "model": "...", "device": "cuda|cpu", "cuda_available": bool}`

### Files to Create

**`services/llm/requirements.txt`**:
```
fastapi==0.115.6
uvicorn[standard]==0.32.1
torch==2.5.1
transformers==4.47.1
accelerate==1.2.1
huggingface-hub==0.27.0
```

**`services/llm/main.py`**:
```python
import os
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = None
tokenizer = None
device = None
executor = ThreadPoolExecutor(max_workers=1)

MODEL_NAME = "cartesia-ai/Llamba-8B"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[startup] Loading {MODEL_NAME} on {device} dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print("[startup] Model loaded.")
    yield
    del model, tokenizer


app = FastAPI(title="LLM Service", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512


class GenerateResponse(BaseModel):
    response: str


def _run_inference(prompt: str, max_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "device": device,
            "cuda_available": torch.cuda.is_available()}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            executor, _run_inference, request.prompt, request.max_tokens
        )
        return GenerateResponse(response=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**`services/llm/Dockerfile`**:
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# HF_TOKEN is required at BUILD TIME because the LLaMA tokenizer files are
# gated and cannot be downloaded without authentication.
# Pass it with: docker build --build-arg HF_TOKEN=hf_xxx ...
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Pre-download tokenizer (and model weights) during build so the container
# starts instantly at runtime without needing network access or the token.
# Weights are cached in /root/.cache/huggingface inside the image layer.
RUN python3 -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('cartesia-ai/Llamba-8B'); \
AutoModelForCausalLM.from_pretrained('cartesia-ai/Llamba-8B', torch_dtype='auto') \
"

# Clear the token from the environment after download (do not leak into runtime)
ENV HF_TOKEN=""

COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> **Note on build-time token**: `HF_TOKEN` is required at **build time** (not just runtime) because `cartesia-ai/Llamba-8B` uses the LLaMA tokenizer, whose config files are gated on HuggingFace and cannot be fetched without authentication — even for the tokenizer alone. The model weights are baked into the image during build. At runtime, no `HF_TOKEN` is needed since all files are already cached in the image.
>
> The token is cleared from the image environment after the download step. However, note that `docker history` may still expose the ARG value if the layer is inspected. For production, use a multi-stage build or a secrets mount (`RUN --mount=type=secret,id=hf_token ...`) instead of `ARG`.

### Verification
```bash
docker build --build-arg HF_TOKEN=<your_token> -t llm-service:local .
docker run --rm -p 8001:8000 llm-service:local
curl http://localhost:8001/api/health
curl -X POST http://localhost:8001/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":50}'
```

---

## Stage 3: Chatbot Backend (FastAPI)

**Prerequisites**: Stage 1 complete. Stage 2 API contract known (documented above).

### Files to Create

**`services/chatbot/backend/requirements.txt`**:
```
fastapi==0.115.6
uvicorn[standard]==0.32.1
httpx==0.28.1
```

**`services/chatbot/backend/main.py`**:
```python
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8001")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Chatbot Backend")


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    url = f"{LLM_SERVICE_URL}/api/generate"
    payload = {"prompt": request.message, "max_tokens": request.max_tokens}
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="LLM service timed out")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"LLM returned {e.response.status_code}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Cannot reach LLM: {e}")
    return ChatResponse(response=resp.json().get("response", ""))


@app.get("/health")
async def health():
    return {"status": "ok", "llm_service_url": LLM_SERVICE_URL}


# Serve React SPA — must be LAST
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR / "static")), name="static")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404, detail="Frontend not built")
        return FileResponse(str(index))
```

> **Note on static files**: React's `build/` directory layout is `build/index.html` + `build/static/{js,css}`. The Dockerfile copies `build/` to `/app/static/`. So `STATIC_DIR = Path(__file__).parent / "static"` points to the React build root, and `STATIC_DIR / "static"` contains the JS/CSS assets.

**`services/chatbot/backend/.env.example`**:
```
LLM_SERVICE_URL=http://llm-service:8000
```

### Verification
```bash
cd services/chatbot/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
LLM_SERVICE_URL=http://localhost:8001 uvicorn main:app --reload --port 8000
curl http://localhost:8000/health
```

---

## Stage 4: Chatbot Frontend (React)

**Prerequisites**: Stage 1 complete. Stage 3 API contract known:
- `POST /chat` body: `{"message": "...", "max_tokens": 512}` → `{"response": "..."}`

Also creates the **Chatbot Dockerfile** (multi-stage: Node builds React → Python serves it).

### Files to Create

**`services/chatbot/frontend/package.json`**:
```json
{
  "name": "chatbot-frontend",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "proxy": "http://localhost:8000",
  "eslintConfig": { "extends": ["react-app"] },
  "browserslist": {
    "production": [">0.2%", "not dead", "not op_mini all"],
    "development": ["last 1 chrome version", "last 1 firefox version", "last 1 safari version"]
  }
}
```

**`services/chatbot/frontend/public/index.html`**:
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Chatbot</title>
  </head>
  <body>
    <noscript>You need JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
```

**`services/chatbot/frontend/src/index.js`**:
```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<React.StrictMode><App /></React.StrictMode>);
```

**`services/chatbot/frontend/src/index.css`**:
```css
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background-color: #f0f2f5;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
}
```

**`services/chatbot/frontend/src/App.js`**:
```javascript
import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, loading]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setInput('');
    setLoading(true);
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, max_tokens: 512 }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  return (
    <div className="chat-container">
      <div className="chat-header"><h1>AI Chatbot</h1></div>
      <div className="messages-area">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-bubble">{msg.content}</div>
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <div className="message-bubble loading">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="input-area">
        <textarea value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
          rows={2} disabled={loading} />
        <button onClick={sendMessage} disabled={loading || !input.trim()}>
          {loading ? '...' : 'Send'}
        </button>
      </div>
    </div>
  );
}

export default App;
```

**`services/chatbot/frontend/src/App.css`**:
```css
.chat-container {
  width: 100%; max-width: 800px; height: 90vh; background: white;
  border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.12);
  display: flex; flex-direction: column; overflow: hidden;
}
.chat-header { padding: 16px 24px; background: #1a73e8; color: white; }
.chat-header h1 { font-size: 1.25rem; font-weight: 600; }
.messages-area { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 12px; }
.message { display: flex; }
.message.user { justify-content: flex-end; }
.message.assistant { justify-content: flex-start; }
.message-bubble {
  max-width: 70%; padding: 10px 16px; border-radius: 18px;
  line-height: 1.5; white-space: pre-wrap; word-break: break-word;
}
.message.user .message-bubble { background: #1a73e8; color: white; border-bottom-right-radius: 4px; }
.message.assistant .message-bubble { background: #f0f2f5; color: #1c1e21; border-bottom-left-radius: 4px; }
.loading { display: flex; align-items: center; gap: 4px; padding: 14px 18px; }
.dot { width: 8px; height: 8px; background: #90949c; border-radius: 50%; animation: bounce 1.2s infinite; }
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-6px); } }
.input-area { padding: 16px; border-top: 1px solid #e4e6eb; display: flex; gap: 8px; align-items: flex-end; }
.input-area textarea {
  flex: 1; resize: none; border: 1px solid #ccd0d5; border-radius: 20px;
  padding: 10px 16px; font-size: 0.95rem; font-family: inherit; outline: none; line-height: 1.4;
}
.input-area textarea:focus { border-color: #1a73e8; }
.input-area button {
  padding: 10px 20px; background: #1a73e8; color: white; border: none;
  border-radius: 20px; cursor: pointer; font-size: 0.95rem; font-weight: 600;
}
.input-area button:disabled { background: #bcc5d1; cursor: not-allowed; }
```

**`services/chatbot/Dockerfile`** (multi-stage; build context = `services/chatbot/`):
```dockerfile
# Stage 1: Build React
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
# Use npm install if package-lock.json does not exist yet
RUN npm install --legacy-peer-deps
COPY frontend/ ./
RUN npm run build

# Stage 2: Python FastAPI
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/main.py .
# React build → /app/static/ (FastAPI looks here via STATIC_DIR)
COPY --from=frontend-builder /app/frontend/build/ ./static/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> **Important**: When building with docker-compose or `docker build`, the build **context** must be `services/chatbot/` (not the repo root). The Dockerfile references `frontend/` and `backend/` as relative paths within that context.

### Verification
```bash
cd services/chatbot
docker build -t chatbot:local .
docker run --rm -e LLM_SERVICE_URL=http://host.docker.internal:8001 -p 3000:8000 chatbot:local
# Open http://localhost:3000 — chat UI must load
curl http://localhost:3000/health
```

---

## Stage 5: Kubernetes Manifests

**Prerequisites**: Stages 2, 3, 4 complete. Agent must know:
- Chatbot image: `<YOUR_ECR_REPO>/chatbot:latest` (port 8000, health: `GET /health`)
- LLM image: `<YOUR_ECR_REPO>/llm:latest` (port 8000, health: `GET /api/health`)
- GPU nodes labeled `role=gpu-nodes`; app nodes labeled `role=app-nodes` (set by Terraform)

All manifests go in `k8s/`.

### Files to Create

**`k8s/namespace.yaml`**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: chatbot
  labels:
    name: chatbot
```

**`k8s/configmap.yaml`**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chatbot-config
  namespace: chatbot
data:
  LLM_SERVICE_URL: "http://llm-service:8000"
```

**`k8s/llm-deployment.yaml`**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service
  namespace: chatbot
  labels:
    app: llm-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-service
  template:
    metadata:
      labels:
        app: llm-service
    spec:
      nodeSelector:
        role: gpu-nodes
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
      containers:
        - name: llm-service
          image: <YOUR_ECR_REPO>/llm:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
          # No HF_TOKEN env var needed at runtime — model is baked into the image at build time.
          # The hf-token-secret is only needed during the image build (CI/CD pipeline),
          # not injected into the running pod.
          resources:
            requests:
              memory: "16Gi"
              cpu: "2"
              nvidia.com/gpu: "1"
            limits:
              memory: "24Gi"
              cpu: "4"
              nvidia.com/gpu: "1"
          livenessProbe:
            httpGet:
              path: /api/health
              port: 8000
            initialDelaySeconds: 300
            periodSeconds: 30
            failureThreshold: 5
          readinessProbe:
            httpGet:
              path: /api/health
              port: 8000
            initialDelaySeconds: 300
            periodSeconds: 10
            failureThreshold: 10
```

**`k8s/llm-service.yaml`**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-service
  namespace: chatbot
spec:
  type: ClusterIP
  selector:
    app: llm-service
  ports:
    - name: http
      port: 8000
      targetPort: 8000
```

**`k8s/chatbot-deployment.yaml`**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chatbot
  namespace: chatbot
  labels:
    app: chatbot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: chatbot
  template:
    metadata:
      labels:
        app: chatbot
    spec:
      nodeSelector:
        role: app-nodes
      containers:
        - name: chatbot
          image: <YOUR_ECR_REPO>/chatbot:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: chatbot-config
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 20
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
```

**`k8s/chatbot-service.yaml`**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: chatbot-service
  namespace: chatbot
spec:
  type: ClusterIP
  selector:
    app: chatbot
  ports:
    - name: http
      port: 8000
      targetPort: 8000
```

**`k8s/ingress.yaml`**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: chatbot-ingress
  namespace: chatbot
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/load-balancer-attributes: idle_timeout.timeout_seconds=120
    alb.ingress.kubernetes.io/healthcheck-path: /health
spec:
  rules:
    - http:
        paths:
          # /api MUST come before / (more-specific prefix first)
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: llm-service
                port:
                  number: 8000
          - path: /
            pathType: Prefix
            backend:
              service:
                name: chatbot-service
                port:
                  number: 8000
```

**`k8s/nvidia-device-plugin.yaml`**:
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  updateStrategy:
    type: RollingUpdate
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      nodeSelector:
        role: gpu-nodes
      priorityClassName: "system-node-critical"
      containers:
        - image: nvcr.io/nvidia/k8s-device-plugin:v0.17.0
          name: nvidia-device-plugin
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
          volumeMounts:
            - name: device-plugin
              mountPath: /var/lib/kubelet/device-plugins
      volumes:
        - name: device-plugin
          hostPath:
            path: /var/lib/kubelet/device-plugins
```

**`k8s/hf-token-secret.yaml.example`** — No longer needed at runtime. The `HF_TOKEN` is consumed only during `docker build` (see Stage 2 and the ECR push steps in Stage 7). Do **not** create this secret in the cluster; the LLM pod does not use it.

**`k8s/kustomization.yaml`**:
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - namespace.yaml
  - configmap.yaml
  - llm-deployment.yaml
  - llm-service.yaml
  - chatbot-deployment.yaml
  - chatbot-service.yaml
  - ingress.yaml
  - nvidia-device-plugin.yaml
```

### Verification
```bash
kubectl apply --dry-run=client -f k8s/
```

---

## Stage 6: Terraform Infrastructure

**Prerequisites**: Stage 1 complete. AWS CLI configured, Terraform >= 1.9.0 installed.

### Key Decisions
- Modules: `terraform-aws-modules/vpc/aws` v5.16.0, `terraform-aws-modules/eks/aws` v20.31.0
- Kubernetes 1.31, region `us-east-1` by default
- `app-nodes`: `t3.medium` (2 replicas), `gpu-nodes`: `g4dn.xlarge` (1 replica, `ami_type = "AL2_x86_64_GPU"`)
- GPU taint: `nvidia.com/gpu=true:NoSchedule` — K8s manifests include matching toleration
- Elastic IPs provisioned (1 per public subnet) for potential NLB static IP binding
- AWS Load Balancer Controller installed via Helm + IRSA

### Files to Create

**`infra/versions.tf`**:
```hcl
terraform {
  required_version = ">= 1.9.0"
  required_providers {
    aws        = { source = "hashicorp/aws",        version = "~> 5.80" }
    kubernetes = { source = "hashicorp/kubernetes",  version = "~> 2.35" }
    helm       = { source = "hashicorp/helm",        version = "~> 2.17" }
    tls        = { source = "hashicorp/tls",         version = "~> 4.0"  }
  }
}
```

**`infra/variables.tf`**:
```hcl
variable "aws_region"   { type = string; default = "us-east-1" }
variable "cluster_name" { type = string; default = "k8s-ssm"   }
variable "environment"  { type = string; default = "production" }
```

**`infra/main.tf`**:
```hcl
provider "aws" { region = var.aws_region }

data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" { state = "available" }

locals {
  common_tags = {
    Project     = var.cluster_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.16.0"

  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"
  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true
  enable_dns_support   = true

  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                    = "1"
  }
  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"           = "1"
  }
  tags = local.common_tags
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "20.31.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.31"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  cluster_endpoint_public_access = true
  enable_irsa                    = true

  eks_managed_node_groups = {
    app-nodes = {
      instance_types = ["t3.medium"]
      min_size = 1; max_size = 4; desired_size = 2
      disk_size = 50
      labels = { role = "app-nodes" }
      tags = merge(local.common_tags, {
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
        "k8s.io/cluster-autoscaler/enabled"             = "true"
      })
    }
    gpu-nodes = {
      instance_types = ["g4dn.xlarge"]
      min_size = 0; max_size = 2; desired_size = 1
      disk_size = 100
      ami_type  = "AL2_x86_64_GPU"
      labels = { role = "gpu-nodes" }
      taints = [{ key = "nvidia.com/gpu", value = "true", effect = "NO_SCHEDULE" }]
      tags = merge(local.common_tags, {
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
        "k8s.io/cluster-autoscaler/enabled"             = "true"
      })
    }
  }
  tags = local.common_tags
}

# IRSA for ALB controller
data "aws_iam_policy_document" "alb_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:sub"
      values   = ["system:serviceaccount:kube-system:aws-load-balancer-controller"]
    }
  }
}

resource "aws_iam_role" "alb_controller" {
  name               = "${var.cluster_name}-alb-controller"
  assume_role_policy = data.aws_iam_policy_document.alb_assume.json
  tags               = local.common_tags
}

resource "aws_iam_policy" "alb_controller" {
  name   = "${var.cluster_name}-alb-controller-policy"
  policy = file("${path.module}/alb-controller-iam-policy.json")
}

resource "aws_iam_role_policy_attachment" "alb_controller" {
  policy_arn = aws_iam_policy.alb_controller.arn
  role       = aws_iam_role.alb_controller.name
}

# Elastic IPs for NLB static IPs (1 per public subnet)
resource "aws_eip" "nlb_eip" {
  count  = length(module.vpc.public_subnets)
  domain = "vpc"
  tags   = merge(local.common_tags, { Name = "${var.cluster_name}-nlb-eip-${count.index}" })
}
```

**`infra/helm.tf`**:
```hcl
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

resource "helm_release" "aws_load_balancer_controller" {
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  version    = "1.10.0"
  namespace  = "kube-system"

  set { name = "clusterName";                                                     value = module.eks.cluster_name }
  set { name = "serviceAccount.create";                                           value = "true" }
  set { name = "serviceAccount.name";                                             value = "aws-load-balancer-controller" }
  set { name = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn";      value = aws_iam_role.alb_controller.arn }

  depends_on = [module.eks]
}
```

**`infra/outputs.tf`**:
```hcl
output "cluster_name"       { value = module.eks.cluster_name }
output "cluster_endpoint"   { value = module.eks.cluster_endpoint }
output "kubeconfig_command" { value = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}" }
output "vpc_id"             { value = module.vpc.vpc_id }
output "nlb_elastic_ips"    { value = aws_eip.nlb_eip[*].public_ip }
output "alb_controller_role_arn" { value = aws_iam_role.alb_controller.arn }
```

**`infra/example.tfvars`**:
```hcl
aws_region   = "us-east-1"
cluster_name = "k8s-ssm"
environment  = "production"
```

**`infra/alb-controller-iam-policy.json`** — Download from official source:
```bash
curl -o infra/alb-controller-iam-policy.json \
  https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.10.0/docs/install/iam_policy.json
```

### Verification
```bash
cd infra
terraform init
terraform validate
terraform plan -var-file=example.tfvars   # ~50-70 resources expected
# terraform apply -var-file=example.tfvars  # takes 15-25 min
```

---

## Stage 7: Local Dev Setup & Running Instructions

**Prerequisites**: All other stages complete.

### Files to Create / Update

**`docker-compose.yml`** (repo root):
```yaml
version: "3.9"

services:
  chatbot:
    build:
      context: services/chatbot
      dockerfile: Dockerfile
    image: chatbot:local
    container_name: chatbot
    ports:
      - "3000:8000"
    environment:
      - LLM_SERVICE_URL=http://llm:8000
    networks:
      - k8s-ssm-net
    restart: unless-stopped

  llm:
    build:
      context: services/llm
      dockerfile: Dockerfile
      args:
        # HF_TOKEN is required at build time for the LLaMA tokenizer download.
        # Ensure HF_TOKEN is set in your shell or .env before running docker compose build.
        HF_TOKEN: ${HF_TOKEN}
    image: llm-service:local
    container_name: llm
    profiles:
      - llm
    ports:
      - "8001:8000"
    environment:
      # No HF_TOKEN needed at runtime — model is baked into the image at build time.
      []
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 300s
    networks:
      - k8s-ssm-net
    restart: unless-stopped
    volumes:
      - hf-cache:/root/.cache/huggingface

networks:
  k8s-ssm-net:
    driver: bridge

volumes:
  hf-cache:
```

**`.env.example`** (repo root):
```bash
# Copy to .env — never commit .env
HF_TOKEN=hf_your_token_here
```

**`README.md`** (replace Stage 1 placeholder with full content):
```markdown
# k8s-ssm

A Kubernetes-deployed chatbot system powered by `cartesia-ai/Llamba-8B`.

## Architecture

```
Browser → /         → Chatbot (FastAPI + React) → POST /chat
Browser → /api/*    → LLM Service (FastAPI + Llamba-8B)
```

## Prerequisites

- Docker + Docker Compose v2
- (GPU inference) NVIDIA Docker runtime (`nvidia-container-toolkit`)
- HuggingFace account with access to `cartesia-ai/Llamba-8B` and an API token

## Quick Start — Full Stack (GPU required)

```bash
cp .env.example .env          # then set HF_TOKEN in .env
docker compose --profile llm up --build
open http://localhost:3000
```

## Quick Start — Chatbot Only (no GPU)

```bash
docker compose up --build     # LLM calls will fail gracefully without the llm service
```

## Hot-Reload Development

```bash
# Terminal 1 (optional GPU service)
docker compose --profile llm up llm

# Terminal 2 — FastAPI backend with auto-reload
cd services/chatbot/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
LLM_SERVICE_URL=http://localhost:8001 uvicorn main:app --reload --port 8000

# Terminal 3 — React dev server (proxies /chat → localhost:8000)
cd services/chatbot/frontend
npm install
npm start    # opens http://localhost:3000 with hot reload
```

## Project Structure

```
k8s-ssm/
├── services/
│   ├── chatbot/
│   │   ├── backend/          FastAPI backend
│   │   ├── frontend/         React SPA
│   │   └── Dockerfile        Multi-stage build
│   └── llm/
│       ├── main.py           FastAPI + HuggingFace inference
│       ├── requirements.txt
│       └── Dockerfile        CUDA 12.1 base image
├── k8s/                      Kubernetes manifests
├── infra/                    Terraform (AWS EKS)
├── plans/                    Implementation plans
├── docker-compose.yml
└── README.md
```

## Deploying to AWS EKS

```bash
# 1. Provision infrastructure (~20 min)
cd infra
curl -o alb-controller-iam-policy.json \
  https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.10.0/docs/install/iam_policy.json
terraform init && terraform apply -var-file=example.tfvars

# 2. Configure kubectl
aws eks update-kubeconfig --region us-east-1 --name k8s-ssm

# 3. Create ECR repos and push images
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

aws ecr create-repository --repository-name chatbot --region $REGION
aws ecr create-repository --repository-name llm-service --region $REGION

aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# Build and push chatbot
docker build -t chatbot:latest services/chatbot/
docker tag chatbot:latest $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/chatbot:latest
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/chatbot:latest

# Build and push LLM service
docker build -t llm-service:latest services/llm/
docker tag llm-service:latest $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/llm-service:latest
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/llm-service:latest

# 4. Update image refs in k8s manifests
sed -i "s|<YOUR_ECR_REPO>/chatbot:latest|$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/chatbot:latest|g" k8s/chatbot-deployment.yaml
sed -i "s|<YOUR_ECR_REPO>/llm:latest|$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/llm-service:latest|g" k8s/llm-deployment.yaml

# 5. Build LLM image with HF_TOKEN as build arg (token needed at build time only)
docker build --build-arg HF_TOKEN=$HF_TOKEN -t llm-service:latest services/llm/
docker tag llm-service:latest $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/llm-service:latest
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/llm-service:latest

# 6. Deploy everything (no HF_TOKEN secret needed in the cluster)
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/

# 7. Get the ALB DNS name (may take 2-3 min to provision)
kubectl get ingress -n chatbot -w
```

## Cost Estimate (us-east-1)

| Resource | Type | Monthly |
|---|---|---|
| EKS Control Plane | — | ~$73 |
| app-nodes (x2) | t3.medium | ~$60 |
| gpu-nodes (x1) | g4dn.xlarge | ~$380 |
| NAT Gateway | — | ~$32 |
| **Total** | | **~$545** |

Set `gpu-nodes desired_size = 0` in Terraform when not in use to save ~$380/month.

## Teardown

```bash
kubectl delete namespace chatbot
terraform destroy -var-file=example.tfvars
```

## Environment Variables

| Variable | Service | Default | Description |
|---|---|---|---|
| `LLM_SERVICE_URL` | chatbot | `http://localhost:8001` | URL of LLM service |
| `HF_TOKEN` | llm | (required) | HuggingFace API token |
```

### Verification (full stack)
1. `cp .env.example .env` → set real `HF_TOKEN`
2. `docker compose --profile llm up --build`
3. Wait for `[startup] Model loaded.` in LLM logs (~5 min)
4. `curl http://localhost:8001/api/health` → `{"status":"ok",...}`
5. `curl http://localhost:3000/health` → `{"status":"ok",...}`
6. Open `http://localhost:3000` → chat UI loads, send "Hello!" → response arrives

---

## Critical Notes for All Agents

1. **Gated model — build-time token required**: `cartesia-ai/Llamba-8B` uses the LLaMA tokenizer, whose files are gated on HuggingFace. `HF_TOKEN` must be provided at **Docker build time** via `--build-arg HF_TOKEN=...` (or `args:` in docker-compose). The token is not needed at runtime because the tokenizer and model weights are baked into the image during the build. Grant your HuggingFace account access to the model at huggingface.co/cartesia-ai/Llamba-8B before building.

2. **GPU availability**: Without CUDA, inference falls back to CPU — expect minutes per response. For local dev without GPU, mock the LLM service or stub the `/api/generate` endpoint.

3. **Static IP caveat**: AWS ALB does not support Elastic IPs directly. The `aws_eip` resources in Terraform are pre-allocated for a potential NLB layer in front of the ALB. For a truly static IP, either: (a) use AWS Global Accelerator (additional cost), or (b) provision an NLB in TCP/443 mode in front of the ALB. The ALB DNS name is stable within the cluster's lifetime and sufficient for most labs.

4. **package-lock.json**: The chatbot Dockerfile uses `npm install --legacy-peer-deps`. After the first `npm install` succeeds, commit the generated `package-lock.json` and switch to `npm ci --legacy-peer-deps` for reproducible builds.

5. **Terraform state**: Plan uses local state. For shared use, add an S3 backend to `versions.tf` with a DynamoDB lock table.

6. **Image placeholders**: K8s manifests use `<YOUR_ECR_REPO>/...` — replace with real ECR URIs before applying to EKS (see deployment instructions above).

7. **GPU taint**: `gpu-nodes` have taint `nvidia.com/gpu=true:NoSchedule`. The LLM deployment includes the matching toleration. No other workloads will land on GPU nodes.
