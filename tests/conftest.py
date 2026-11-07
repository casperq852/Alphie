import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Isolated data dir + no external services; must happen before app modules are imported.
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="alphie_test_")
os.environ["OPENAI_API_KEY"] = "test"
for var in ("USE_QDRANT", "QDRANT_URL", "TAVILY_API_KEY"):
    os.environ.pop(var, None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import company_normalizer  # noqa: E402
import llm  # noqa: E402
import rag  # noqa: E402


def fake_embed(texts, batch_size=64):
    """Deterministic bag-of-words hashing embedding: shared words -> high cosine similarity."""
    out = np.zeros((len(texts), llm.EMB_DIM), dtype="float32")
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % llm.EMB_DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-12, None)


@pytest.fixture
def offline(monkeypatch):
    """Patch every OpenAI touch point."""
    monkeypatch.setattr(rag, "embed", fake_embed)
    monkeypatch.setattr(company_normalizer, "embed", fake_embed)

    def fake_chat(messages, model=None, temperature=None, max_tokens=None):
        return "[]"

    monkeypatch.setattr(company_normalizer, "chat", fake_chat)
    monkeypatch.setattr(rag, "chat", fake_chat)
    return rag
