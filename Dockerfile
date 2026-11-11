# Mini-GPT web app + REST API, CPU-only image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    MODEL_DIR=/app/model

# Run as an unprivileged user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# CPU build of PyTorch: ~200 MB instead of several GB with CUDA
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY --chown=appuser model.py tokenizer.py inference.py api.py app.py server.py ./
# The model folder is created by `python export.py --run runs/<run>`.
# If it's missing, set MODEL_REPO so the app downloads it from the Hugging Face Hub at startup.
COPY --chown=appuser model/ ./model/

USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health')" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
