import json
import re
import numpy as np
from typing import List
from openai import OpenAI
from pathlib import Path

client = OpenAI()

# --------------------------------------------
# TEXT CLEANER
# --------------------------------------------
def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# --------------------------------------------
# EMBEDDING-BASED MATCHING
# --------------------------------------------
_cached_embs = None
_cached_names = None

def load_company_index(company_list: List[str], emb_model: str = "text-embedding-3-large"):
    """
    Build or update a small local embedding index of known company names.
    """
    global _cached_embs, _cached_names
    if not company_list:
        _cached_embs = np.zeros((0, 1536), dtype="float32")
        _cached_names = []
        return

    # Encode names into embeddings
    resp = client.embeddings.create(model=emb_model, input=company_list)
    embs = np.array([d.embedding for d in resp.data], dtype="float32")

    # Normalize for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-12, np.inf)

    _cached_embs = embs
    _cached_names = company_list


def resolve_by_embedding(name: str) -> str:
    """
    If we have an embedding index of company names from the PDF store,
    match the input name to the closest canonical name.
    """
    global _cached_embs, _cached_names

    if _cached_embs is None or _cached_embs.shape[0] == 0:
        return name  # no index loaded

    # Embed the query
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=[name]
    )
    vec = np.array(resp.data[0].embedding, dtype="float32")
    vec = vec / max(1e-12, np.linalg.norm(vec))

    sims = _cached_embs @ vec
    idx = int(np.argmax(sims))
    return _cached_names[idx]


# --------------------------------------------
# LLM-BASED CANONICALIZATION
# --------------------------------------------
def llm_canonicalize(raw_names: List[str]) -> List[str]:
    """
    Use an LLM to map variations ("jd sports", "jd sports group", "jd sports plc")
    into a single canonical form.
    """
    if not raw_names:
        return []

    prompt = f"""
Zet elke bedrijfsnaam hieronder om naar de meest waarschijnlijke officiële,
unieke en volledige bedrijfsnaam. Normaliseer spelling, afkortingen, en 
ontbrekende delen.

Geef uitsluitend een geldige Python-lijst van strings terug.

Bedrijfsnamen:
{raw_names}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return raw_names  # fallback


# --------------------------------------------
# MASTER RESOLVER
# --------------------------------------------
def canonicalize(raw_names: List[str]) -> List[str]:
    """
    Unified resolver:
    1. LLM normalizes spelling/structure
    2. Embedding index aligns to known PDF companies
    """
    if not raw_names:
        return []

    # Step 1 — Clean duplicates before LLM
    cleaned = sorted(set(raw_names), key=lambda x: x.lower())

    # Step 2 — LLM canonicalize
    llm_names = llm_canonicalize(cleaned)

    # Step 3 — Embedding-based mapping to local PDF companies
    resolved = [resolve_by_embedding(n) for n in llm_names]

    # Step 4 — Remove duplicates again
    return sorted(set(resolved), key=lambda x: x.lower())


def _extract_company_names_from_excel(path: Path) -> List[str]:
    raw_candidates = []

    try:
        xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except Exception:
        return []

    # 1. filename is always a hint
    raw_candidates.append(path.stem)

    for sheet_name, df in xls.items():
        # 2. sheet names are often close to the company name
        raw_candidates.append(sheet_name)

        if df is None or df.empty:
            continue

        # 3. Look at first column (top 10 rows)
        if df.shape[0] > 0:
            first_col = df.columns[0]
            for val in df[first_col].head(10).astype(str).tolist():
                if isinstance(val, str) and val.strip():
                    raw_candidates.append(val)

        # 4. Look at first row (top 10 columns)
        if df.shape[1] > 1:
            for val in df.iloc[0, :10].astype(str).tolist():
                if isinstance(val, str) and val.strip():
                    raw_candidates.append(val)

    # Clean weird strings and keep only strings with letters
    cleaned = [
        s for s in raw_candidates
        if any(c.isalpha() for c in s)
    ]

    # Remove extreme garbage
    cleaned = [s for s in cleaned if len(s) <= 80]

    return canonicalize(cleaned)
