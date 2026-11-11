"""
Production entry point: the REST API and the web app in one service.

    python server.py                         # uses ./model (created by export.py)
    python server.py --model runs/shakespeare_char_small
    uvicorn server:app --host 0.0.0.0 --port 7860   # how Docker / Hugging Face run it

    Web app:   http://localhost:7860/
    API docs:  http://localhost:7860/docs

Environment variables:
    MODEL_DIR        model folder (default: model)
    MODEL_REPO       Hugging Face model repo to download if MODEL_DIR is missing
    MAX_CONCURRENT   simultaneous generations (default 2)
    MAX_NEW_TOKENS   upper limit per request (default 1000)
    TORCH_THREADS    CPU threads for PyTorch (default: all)
"""
import argparse
import logging
import os
import time

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api import build_router
from app import CSS, THEME, build_ui
from inference import Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mini-gpt")


def create_app(model_dir=None) -> FastAPI:
    threads = int(os.environ["TORCH_THREADS"]) if os.environ.get("TORCH_THREADS") else None
    engine = Engine(model_dir, threads=threads)
    info = engine.info()
    log.info("Loaded %s: %.2fM params, %d layers", info["name"], info["parameters"] / 1e6, info["n_layer"])

    api = FastAPI(
        title="Mini-GPT API",
        version="1.0.0",
        description="Text generation and interpretability endpoints for a GPT model built from scratch.",
    )
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    @api.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/api"):
            log.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code,
                     (time.perf_counter() - start) * 1000)
        return response

    api.include_router(build_router(engine))

    demo = build_ui(engine)
    demo.queue(default_concurrency_limit=int(os.environ.get("MAX_CONCURRENT", 2)), max_size=32)
    return gr.mount_gradio_app(api, demo, path="/", theme=THEME, css=CSS, footer_links=["api"])


app = create_app() if os.environ.get("MINI_GPT_NO_AUTOLOAD") != "1" and __name__ != "__main__" else None

if __name__ == "__main__":
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="model folder or training run folder")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    args = p.parse_args()
    uvicorn.run(create_app(args.model), host=args.host, port=args.port)
