# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System deps (FAISS + wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    # canonical storage path inside the container (matches your code)
    DATA_DIR=/app/data

WORKDIR /app

# Copy source
COPY . /app

# Install deps used in your repo
RUN pip install --upgrade pip && pip install \
    streamlit \
    python-dotenv \
    openai~=1.52 \
    tiktoken \
    numpy \
    pypdf \
    faiss-cpu \
    qdrant-client \
    requests \
    dateparser \
    python-dateutil

EXPOSE 8501

# Basic healthcheck for Streamlit
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8588/_stcore/health || exit 1

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false
ENV STREAMLIT_SERVER_RUN_ON_SAVE=false
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=100

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
