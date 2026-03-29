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

# When ENV=dev, use the lighter 1B model for local smoke testing.
# In all other environments the full 8B model is used.
ENV = os.getenv("ENV", "")
MODEL_NAME = "cartesia-ai/Llamba-1B" if ENV == "dev" else "cartesia-ai/Llamba-8B"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[startup] ENV={ENV!r} — loading {MODEL_NAME} on {device} dtype={dtype}")
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
