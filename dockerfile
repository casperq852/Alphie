# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps (FAISS + wheels + healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false \
    STREAMLIT_SERVER_RUN_ON_SAVE=false \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=100 \
    DATA_DIR=/app/data

WORKDIR /app

# Install python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy source
COPY . /app

# Non-root runtime user
RUN useradd -m -u 10001 appuser \
 && mkdir -p /app/data \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
