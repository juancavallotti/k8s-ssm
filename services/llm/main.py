import os
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel

model = None
tokenizer = None
device = None
executor = ThreadPoolExecutor(max_workers=1)

# When ENV=dev, use the lighter 1B model for local smoke testing.
# In all other environments the full 8B model is used.
ENV = os.getenv("ENV", "")

if ENV == "dev":
    MODEL_NAME = "cartesia-ai/Llamba-1B"
    TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
else:
    MODEL_NAME = "cartesia-ai/Llamba-8B"
    TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[startup] ENV={ENV!r} — loading {MODEL_NAME} on {device} dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = LlambaLMHeadModel.from_pretrained(MODEL_NAME)
    model.to(device=device, dtype=dtype)
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
    inputs = tokenizer(prompt, return_tensors="pt")
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

    # Decode only the newly generated tokens (strip the prompt)
    generated_ids = output.sequences[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "tokenizer": TOKENIZER_NAME,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }


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
