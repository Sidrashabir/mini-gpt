"""
REST API for Mini-GPT.

Endpoints (interactive docs at /docs):
    GET  /api/health            liveness check
    GET  /api/model             model details and metrics
    POST /api/generate          full completion as JSON
    POST /api/generate/stream   completion streamed as Server-Sent Events
    POST /api/next-token        top-k next-token probabilities
    POST /api/attention         attention weights for one layer
"""
import asyncio
import json
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference import Engine

MAX_PROMPT_CHARS = 2000
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 1000))
# CPU servers handle a few generations at once; extra requests wait instead of crashing the box.
_slots = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT", 2)))


class GenerateRequest(BaseModel):
    prompt: str = Field("ROMEO:", max_length=MAX_PROMPT_CHARS, examples=["ROMEO:"])
    max_new_tokens: int = Field(300, ge=1, le=MAX_NEW_TOKENS)
    temperature: float = Field(0.8, gt=0.0, le=2.0)
    top_k: int | None = Field(50, ge=1, le=1000)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    seed: int | None = Field(None, description="set for reproducible output")
    stop: str | None = Field(None, max_length=50, description="stop when this text appears")


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    tokens_generated: int
    seconds: float
    tokens_per_second: float | None


class NextTokenRequest(BaseModel):
    prompt: str = Field(..., max_length=MAX_PROMPT_CHARS, examples=["To be, or not to b"])
    k: int = Field(10, ge=1, le=50)
    temperature: float = Field(1.0, gt=0.0, le=2.0)


class AttentionRequest(BaseModel):
    prompt: str = Field(..., max_length=200, examples=["To be, or not to be"])
    layer: int = Field(0, ge=0)


def build_router(engine: Engine) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["Mini-GPT"])

    def gen_kwargs(req: GenerateRequest):
        return dict(max_new_tokens=req.max_new_tokens, temperature=req.temperature,
                    top_k=req.top_k, top_p=req.top_p, seed=req.seed, stop=req.stop)

    @router.get("/health")
    def health():
        return {"status": "ok"}

    @router.get("/model")
    def model_info():
        return engine.info()

    @router.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        async with _slots:
            return await asyncio.to_thread(engine.generate, req.prompt, **gen_kwargs(req))

    @router.post("/generate/stream")
    async def generate_stream(req: GenerateRequest, request: Request):
        async def events():
            async with _slots:
                it = engine.stream(req.prompt, **gen_kwargs(req))
                while True:
                    if await request.is_disconnected():
                        break
                    piece = await asyncio.to_thread(next, it, None)
                    if piece is None:
                        break
                    yield f"data: {json.dumps({'token': piece})}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    @router.post("/next-token")
    def next_token(req: NextTokenRequest):
        return engine.next_token(req.prompt, k=req.k, temperature=req.temperature)

    @router.post("/attention")
    def attention(req: AttentionRequest):
        if req.layer >= engine.cfg.n_layer:
            raise HTTPException(422, f"layer must be between 0 and {engine.cfg.n_layer - 1}")
        return engine.attention(req.prompt, layer=req.layer)

    return router
