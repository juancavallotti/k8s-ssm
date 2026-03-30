import os
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

ENV = os.getenv("ENV", "")

# Redirect HuggingFace cache to the persistent volume mount.
# Falls back to the default ~/.cache/huggingface when running locally.
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "/model-cache")
os.environ["HF_HOME"] = HF_CACHE_DIR

# Detect backend at import time.
# MLX is available when running natively on Apple Silicon macOS.
# Fall back to pytorch (CUDA or CPU) in all other environments.
try:
    import cartesia_mlx as _cmx  # noqa: F401
    import mlx.core as _mx       # noqa: F401
    BACKEND = "mlx"
except ImportError:
    BACKEND = "pytorch"

if ENV == "dev":
    MODEL_NAME = "cartesia-ai/Llamba-1B-4bit-mlx" if BACKEND == "mlx" else "cartesia-ai/Llamba-1B"
    TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
else:
    MODEL_NAME = "cartesia-ai/Llamba-8B-4bit-mlx" if BACKEND == "mlx" else "cartesia-ai/Llamba-8B"
    TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"

model = None
tokenizer = None  # Not used by the MLX backend (generate() takes a string directly)
device = None
executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device

    if BACKEND == "mlx":
        import cartesia_mlx as cmx
        import mlx.core as mx
        device = "mlx"
        print(f"[startup] ENV={ENV!r} BACKEND=mlx — loading {MODEL_NAME}")
        model = cmx.from_pretrained(MODEL_NAME)
        model.set_dtype(mx.float32)
    else:
        import torch
        from transformers import AutoTokenizer
        from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"[startup] ENV={ENV!r} BACKEND=pytorch — loading {MODEL_NAME} on {device} dtype={dtype}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        model = LlambaLMHeadModel.from_pretrained(
            MODEL_NAME,
            device_map=device,
            torch_dtype=dtype,
        )
        model.eval()

    print("[startup] Model loaded.")
    yield
    del model
    if tokenizer is not None:
        del tokenizer


app = FastAPI(title="LLM Service", lifespan=lifespan)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    messages: list[Message]
    system: str = "You are a helpful assistant."
    max_tokens: int = 512


class GenerateResponse(BaseModel):
    response: str


def _apply_chat_template(messages: list[Message], system: str) -> str:
    """Format a conversation into the Llama 3 instruct chat template."""
    result = (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    )
    for msg in messages:
        result += f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}<|eot_id|>"
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return result


def _run_inference(messages: list[Message], system: str, max_tokens: int) -> str:
    formatted = _apply_chat_template(messages, system)

    if BACKEND == "mlx":
        chunks: list[str] = []
        for text in model.generate(formatted, max_tokens=max_tokens, temperature=0.7, top_p=0.9):
            chunks.append(text)
            if "<|eot_id|>" in "".join(chunks):
                break
        return "".join(chunks).split("<|eot_id|>")[0].strip()

    import torch
    from functools import partial

    inputs = tokenizer(formatted, return_tensors="pt")  # type: ignore[misc]
    input_ids = inputs.input_ids.to(device)
    max_length = input_ids.shape[1] + max_tokens

    generate_fn = partial(
        model.generate,
        cg=True,
        return_dict_in_generate=True,
        output_scores=False,
        enable_timing=False,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.inference_mode():
        output = generate_fn(input_ids=input_ids, max_length=max_length)

    generated_ids = output.sequences[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()


@app.get("/api/health")
async def health():
    info: dict = {
        "status": "ok" if model is not None else "loading",
        "model": MODEL_NAME,
        "device": device,
        "backend": BACKEND,
    }
    if BACKEND == "pytorch":
        import torch
        info["tokenizer"] = TOKENIZER_NAME
        info["cuda_available"] = torch.cuda.is_available()
    return info


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            executor, _run_inference, request.messages, request.system, request.max_tokens
        )
        return GenerateResponse(response=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
