FROM python:3.12-slim@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 AS builder

ENV VIRTUAL_ENV=/venv \
    PATH=/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv "$VIRTUAL_ENV"

COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir torch==2.13.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --requirement requirements-runtime.txt
RUN pip install --no-cache-dir setuptools==84.0.0
RUN mkdir /runtime-libs \
    && cp -L /lib/x86_64-linux-gnu/libffi.so.8 /runtime-libs/libffi.so.8

FROM gcr.io/distroless/cc-debian13:nonroot@sha256:d0b79eb697888ecb8ef019bbb7192e4f41974830ea95f0543123eaaeb2d5fd2c

ENV HOME=/tmp \
    PATH=/venv/bin:/usr/local/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY --from=builder /venv /venv
COPY --from=builder /runtime-libs/libffi.so.8 /lib/x86_64-linux-gnu/libffi.so.8
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
