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
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404, detail="Frontend not built")
        return FileResponse(str(index))
