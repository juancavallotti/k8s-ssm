AWS_REGION   ?= us-west-2
CLUSTER_NAME ?= k8s-ssm
ACCOUNT_ID   := $(shell aws sts get-caller-identity --query Account --output text)
ECR_BASE     := $(ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com

.PHONY: all infra deploy build push setup-k8s render-k8s teardown help

## Full deployment from scratch
all: infra build push setup-k8s deploy

## ── Infrastructure ────────────────────────────────────────────────────────────

# Step 1: provision cluster and core AWS resources (skips helm provider)
infra-base:
	cd infra && terraform init -input=false
	cd infra && terraform apply \
		-target=module.vpc \
		-target=module.eks \
		-target=aws_iam_role.ebs_csi_driver \
		-target=aws_iam_role_policy_attachment.ebs_csi_driver \
		-target=aws_iam_role.alb_controller \
		-target=aws_iam_policy.alb_controller \
		-target=aws_iam_role_policy_attachment.alb_controller \
		-target=aws_eip.nlb_eip \
		-auto-approve
	aws eks update-kubeconfig --region $(AWS_REGION) --name $(CLUSTER_NAME)
	@echo ""
	@echo "Cluster is up. Adding EKS access entry for current IAM identity..."
	$(eval CALLER_ARN := $(shell aws sts get-caller-identity --query Arn --output text))
	aws eks create-access-entry \
		--cluster-name $(CLUSTER_NAME) \
		--principal-arn $(CALLER_ARN) \
		--region $(AWS_REGION) 2>/dev/null || true
	aws eks associate-access-policy \
		--cluster-name $(CLUSTER_NAME) \
		--principal-arn $(CALLER_ARN) \
		--policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \
		--access-scope type=cluster \
		--region $(AWS_REGION) 2>/dev/null || true

# Step 2: install helm chart (ALB controller) — requires cluster to be up
infra-helm:
	cd infra && terraform apply -auto-approve

# Full infra in correct order
infra: infra-base infra-helm

## ── Docker images ─────────────────────────────────────────────────────────────

ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_BASE)

ecr-repos:
	aws ecr describe-repositories --region $(AWS_REGION) --repository-names llm 2>/dev/null || \
		aws ecr create-repository --repository-name llm --region $(AWS_REGION)
	aws ecr describe-repositories --region $(AWS_REGION) --repository-names chatbot 2>/dev/null || \
		aws ecr create-repository --repository-name chatbot --region $(AWS_REGION)

build:
	docker build --platform linux/amd64 -t llm services/llm/
	docker build --platform linux/amd64 -t chatbot services/chatbot/

push: ecr-login ecr-repos
	docker tag llm:latest $(ECR_BASE)/llm:latest
	docker tag chatbot:latest $(ECR_BASE)/chatbot:latest
	docker push $(ECR_BASE)/llm:latest
	docker push $(ECR_BASE)/chatbot:latest

## ── Kubernetes setup ──────────────────────────────────────────────────────────

# Renders deployment manifests from templates by substituting the ECR base URL
render-k8s:
	sed 's|{{ECR_BASE}}|$(ECR_BASE)|g' k8s/chatbot-deployment.yaml.template > k8s/chatbot-deployment.yaml
	sed 's|{{ECR_BASE}}|$(ECR_BASE)|g' k8s/llm-deployment.yaml.template     > k8s/llm-deployment.yaml

# Creates the HF token secret — requires HF_TOKEN env var to be set
setup-k8s:
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "ERROR: HF_TOKEN environment variable is not set."; \
		echo "Run: export HF_TOKEN=hf_your_token_here"; \
		exit 1; \
	fi
	kubectl create secret generic hf-token \
		--from-literal=token=$(HF_TOKEN) \
		--namespace=chatbot \
		--dry-run=client -o yaml | kubectl apply -f -

deploy: render-k8s
	kubectl apply -k k8s/
	@echo ""
	@echo "Waiting for chatbot to be ready..."
	kubectl rollout status deployment/chatbot -n chatbot --timeout=120s
	@echo ""
	@echo "LLM service is starting (model download may take a few minutes on first run)..."
	kubectl rollout status deployment/llm-service -n chatbot --timeout=300s

## ── Teardown ──────────────────────────────────────────────────────────────────

teardown:
	@echo "This will destroy ALL infrastructure. Press Ctrl+C to cancel, Enter to continue."
	@read confirm
	kubectl delete -k k8s/ 2>/dev/null || true
	cd infra && terraform destroy -auto-approve

## ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Usage:"
	@echo "  make all            Full deploy from scratch (infra + build + push + k8s)"
	@echo "  make infra          Provision AWS infrastructure (two-step, handles chicken-and-egg)"
	@echo "  make build          Build Docker images (linux/amd64)"
	@echo "  make push           Push images to ECR (creates repos if needed)"
	@echo "  make render-k8s     Render deployment YAMLs from templates (fills in ECR URL)"
	@echo "  make setup-k8s      Create Kubernetes secrets (requires HF_TOKEN env var)"
	@echo "  make deploy         Render manifests and apply to k8s"
	@echo "  make teardown       Destroy everything"
	@echo ""
	@echo "Required:"
	@echo "  AWS credentials configured (aws configure --profile terraform)"
	@echo "  HF_TOKEN env var set for setup-k8s target"
