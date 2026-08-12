FROM cgr.dev/chainguard/python:3.12-dev AS builder

ENV VIRTUAL_ENV=/venv \
    PATH=/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv "$VIRTUAL_ENV"

COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir --requirement requirements-runtime.txt

FROM cgr.dev/chainguard/python:3.12

ENV HOME=/tmp \
    PATH=/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY --from=builder /venv /venv
COPY app.py ./
COPY deployment/ ./deployment/
COPY models/ ./models/
COPY artifacts/public_v1/ ./artifacts/public_v1/
COPY docs/assets/ ./docs/assets/
COPY evidence/ ./evidence/
COPY examples/public_synthetic_input.json ./examples/public_synthetic_input.json

USER 65532:65532

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

ENTRYPOINT []
CMD ["/venv/bin/python", "-m", "uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
