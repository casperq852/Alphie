"""Central OpenAI access: model config, lazy client, and a chat helper that works for both
classic chat models (gpt-4o-mini) and reasoning models (gpt-5*, o-series)."""

import os
import re
from functools import lru_cache

import numpy as np
from openai import OpenAI

# -------- Model configuration (single source of truth) --------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # answering model
INGEST_MODEL = os.getenv("INGEST_MODEL", os.getenv("MINI_MODEL", "gpt-4o-mini"))  # cheap ingestion-time model
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-large")
EMB_DIM = 3072 if EMB_MODEL == "text-embedding-3-large" else 1536
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")  # minimal | low | medium | high
# Reasoning models spend completion tokens on hidden reasoning; add headroom so answers are not cut off/empty.
REASONING_TOKEN_BUFFER = int(os.getenv("REASONING_TOKEN_BUFFER", "4000"))

AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-5-mini", "gpt-5"]

_REASONING_RE = re.compile(r"^(o\d|gpt-5)", re.IGNORECASE)


def is_reasoning_model(model: str) -> bool:
    """gpt-5 family and o-series reject `temperature`/`max_tokens` and need `max_completion_tokens`."""
    return bool(_REASONING_RE.match((model or "").strip()))


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Created lazily so importing modules (and tests) does not require an API key."""
    return OpenAI()


def build_chat_params(model: str, temperature: float | None, max_tokens: int | None) -> dict:
    params: dict = {"model": model}
    if is_reasoning_model(model):
        if max_tokens:
            params["max_completion_tokens"] = int(max_tokens) + REASONING_TOKEN_BUFFER
        if REASONING_EFFORT:
            params["reasoning_effort"] = REASONING_EFFORT
    else:
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens:
            params["max_tokens"] = int(max_tokens)
    return params


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = 0.2,
    max_tokens: int | None = None,
) -> str:
    """Single chat completion; returns the message text ('' if the model returned nothing)."""
    params = build_chat_params(model or OPENAI_MODEL, temperature, max_tokens)
    resp = get_client().chat.completions.create(messages=messages, **params)
    return (resp.choices[0].message.content or "").strip()


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed texts in batches; returns L2-normalised float32 matrix of shape (n, EMB_DIM)."""
    if not texts:
        return np.zeros((0, EMB_DIM), dtype="float32")
    out = []
    step = max(1, batch_size)
    for i in range(0, len(texts), step):
        resp = get_client().embeddings.create(model=EMB_MODEL, input=texts[i : i + step])
        out.append(np.array([d.embedding for d in resp.data], dtype="float32"))
    vecs = np.vstack(out)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


def strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()
