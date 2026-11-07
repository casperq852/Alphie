"""Company-name canonicalisation: LLM spelling normalisation + embedding match against the
companies already known from ingested documents."""

import json
import os
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from db import DATA_DIR
from llm import EMB_DIM, INGEST_MODEL, chat, embed, strip_code_fences

# Below this cosine similarity a name is NOT mapped onto a known company (prevents silently
# analysing the wrong company when the requested one is not in the repository).
MATCH_THRESHOLD = float(os.getenv("COMPANY_MATCH_THRESHOLD", "0.55"))
_EMB_CACHE_PATH = DATA_DIR / "company_name_embeddings.npz"

_lock = threading.Lock()
_names: list[str] = []
_embs = np.zeros((0, EMB_DIM), dtype="float32")
_disk_cache: dict[str, np.ndarray] | None = None


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_disk_cache() -> dict[str, np.ndarray]:
    global _disk_cache
    if _disk_cache is None:
        _disk_cache = {}
        if _EMB_CACHE_PATH.exists():
            try:
                data = np.load(_EMB_CACHE_PATH, allow_pickle=False)
                for n, v in zip(data["names"].tolist(), data["embs"], strict=True):
                    _disk_cache[str(n)] = v
            except Exception as e:
                print(f"[company_normalizer] ignoring unreadable embedding cache: {e}")
    return _disk_cache


def _save_disk_cache(cache: dict[str, np.ndarray]) -> None:
    if not cache:
        return
    names = list(cache.keys())
    try:
        np.savez(_EMB_CACHE_PATH, names=np.array(names), embs=np.vstack([cache[n] for n in names]))
    except Exception as e:
        print(f"[company_normalizer] could not persist embedding cache: {e}")


def load_company_index(company_list: list[str]) -> None:
    """Build/refresh the in-memory index of known company names.

    Name embeddings are cached on disk, so only names not seen before are sent to the API.
    Calling this repeatedly with an unchanged list is free."""
    global _names, _embs
    names = sorted({c.strip() for c in company_list if c and c.strip()})
    with _lock:
        if names == _names:
            return
        cache = _load_disk_cache()
        missing = [n for n in names if n not in cache]
        if missing:
            for n, v in zip(missing, embed(missing, batch_size=512), strict=True):
                cache[n] = v
            _save_disk_cache(cache)
        _names = names
        _embs = np.vstack([cache[n] for n in names]) if names else np.zeros((0, EMB_DIM), dtype="float32")


def resolve_by_embedding(names: list[str]) -> list[str]:
    """Map each name to the closest known company if similar enough, else keep it unchanged."""
    if not names or _embs.shape[0] == 0:
        return list(names)
    vecs = embed(names)
    sims = vecs @ _embs.T
    out = []
    for name, row in zip(names, sims, strict=True):
        idx = int(np.argmax(row))
        out.append(_names[idx] if float(row[idx]) >= MATCH_THRESHOLD else name)
    return out


def _parse_name_list(content: str) -> list[str] | None:
    content = strip_code_fences(content)
    try:
        data = json.loads(content)
    except Exception:
        return None
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    return None


def llm_canonicalize(raw_names: list[str]) -> list[str]:
    """Normalise variations ("jd sports", "jd sports plc") into one official company name."""
    if not raw_names:
        return []
    prompt = (
        "Zet elke bedrijfsnaam of ticker hieronder om naar de meest waarschijnlijke officiële, unieke en volledige "
        "bedrijfsnaam. Normaliseer spelling, afkortingen en ontbrekende delen. Houd de volgorde en het aantal gelijk.\n"
        "Geef uitsluitend een JSON-array van strings terug, zonder code fences.\n\n"
        f"Bedrijfsnamen:\n{json.dumps(raw_names, ensure_ascii=False)}"
    )
    try:
        parsed = _parse_name_list(chat([{"role": "user", "content": prompt}], model=INGEST_MODEL, temperature=0.0))
    except Exception as e:
        print(f"[company_normalizer] llm_canonicalize failed: {e}")
        parsed = None
    if not parsed or len(parsed) != len(raw_names):
        return list(raw_names)
    return [p.strip() or r for p, r in zip(parsed, raw_names, strict=True)]


def canonicalize(raw_names: list[str]) -> list[str]:
    """1) LLM normalises spelling, 2) embedding match aligns to companies known from ingested docs."""
    cleaned = sorted({n.strip() for n in raw_names if n and n.strip()}, key=str.lower)
    if not cleaned:
        return []
    resolved = resolve_by_embedding(llm_canonicalize(cleaned))
    return sorted(set(resolved), key=str.lower)


# --------------------------------------------
# Company extraction from free text (PDFs, queries)
# --------------------------------------------
# Drug suffixes are only applied to single-word candidates ("pembrolizumab"), never to company names.
_DRUG_SUFFIXES = (
    "mab", "zumab", "ximab", "limab", "tinib", "ciclib", "parib", "setron", "mycin", "floxacin",
    "dazole", "oxetine", "pril", "sartan", "statin", "gliptin", "gliflozin", "prazole", "caine",
    "dronate", "avir", "rudin", "tamide", "terol", "dopa", "prost", "afil",
)  # fmt: skip
_BAD_TOKENS = {"tablet", "tablets", "solution", "capsule", "capsules", "injection", "mg", "ml", "iv", "po", "qd",
               "bid", "tid", "q4w", "q2w", "placebo"}  # fmt: skip
# Research houses / data vendors that are (practically) never the subject of a note. Listed banks that also
# publish research (UBS, ING, HSBC, Barclays, JPMorgan, ...) are deliberately NOT here: they are often the
# subject (e.g. a banks deep dive). The LLM prompt excludes them when they are only the publisher.
_BROKER_VENDORS = {
    "kepler cheuvreux", "berenberg", "bernstein", "cowen", "td cowen", "evercore isi", "piper sandler",
    "cantor fitzgerald", "roth mkm", "redburn", "redburn atlantic", "oddo", "oddo bhf", "kbw", "keefe bruyette",
    "numis", "peel hunt", "liberum", "arete", "wolfe research", "needham", "rosenblatt", "exane", "exane bnp",
    "bofa securities", "rbc capital markets", "factset", "bloomberg", "bloomberg intelligence", "refinitiv",
    "visible alpha", "morningstar", "s&p global market intelligence", "spglobal",
}  # fmt: skip
# Research arm of a bank ("UBS Global Research", "ING Research", "Barclays Equities") -> publisher, not subject.
_RESEARCH_ARM_RE = re.compile(r"\b(research|securities|equities|global markets|investment bank(ing)?)\b", re.I)


def _contains_term(low: str, terms: set[str]) -> bool:
    """Whole-word match, so 'po' does not hit 'Sports'/'Corporation' and 'iv' does not hit 'Givaudan'."""
    return any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low) for t in terms)


def clean_company_list(items: list[str], max_items: int = 8) -> list[str]:
    clean, seen = [], set()
    for x in items:
        s = re.sub(r"\s+", " ", str(x)).strip().strip("\"'")
        if len(s) < 2 or not re.search(r"[A-Za-z]", s) or s.islower():
            continue
        low = s.lower()
        if _contains_term(low, _BAD_TOKENS) or _contains_term(low, _BROKER_VENDORS) or _RESEARCH_ARM_RE.search(s):
            continue
        if " " not in s and any(low.endswith(sfx) for sfx in _DRUG_SUFFIXES):
            continue
        if (re.search(r"[A-Za-z]\d{2,}", s) and "-" in s) or re.fullmatch(r"[A-Z]{1,5}-?\d{3,}[A-Z]?", s):
            # trial / compound codes like "AZD-1234", "MK3475", "LY3298176-x"
            continue
        if low in seen:
            continue
        seen.add(low)
        clean.append(s)
    return clean[:max_items]


def classify_companies(raw_text: str, model: str | None = None) -> list[str]:
    """Extract company names / tickers from a research document or user query."""
    system = (
        "Extract the public companies (canonical names or tickers) that are a SUBJECT of the text: a broker research "
        "PDF or a user question. EXCLUDE drug names, molecules, trial arms, medicine brands, dosages and medical "
        "terms. EXCLUDE the publishing broker / research house / data vendor (e.g. the bank that wrote the note, "
        "Bloomberg, FactSet), unless that institution is itself being analysed. "
        "If a company is mentioned under several names or with a ticker, return each variant separately. "
        "Return ONLY a JSON array of strings, no code fences."
    )
    content = chat(
        [{"role": "system", "content": system}, {"role": "user", "content": f"Text:\n{(raw_text or '')[:8000]}"}],
        model=model or INGEST_MODEL,
        temperature=0.2,
    )
    content = strip_code_fences(content)
    parsed = _parse_name_list(content)
    items = parsed if parsed is not None else [p for p in re.split(r"[,;\n]+", content) if p.strip()]
    return clean_company_list(items)


def _extract_company_names_from_excel(path: Path) -> list[str]:
    try:
        xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except Exception:
        return []

    raw_candidates = [path.stem]  # filename is always a hint
    for sheet_name, df in xls.items():
        raw_candidates.append(str(sheet_name))  # sheet names are often close to the company name
        if df is None or df.empty:
            continue
        raw_candidates.extend(df[df.columns[0]].head(10).astype(str).tolist())  # first column, top rows
        if df.shape[1] > 1:
            raw_candidates.extend(df.iloc[0, :10].astype(str).tolist())  # first row

    cleaned = [s.strip() for s in raw_candidates if any(c.isalpha() for c in s) and len(s) <= 80]
    cleaned = [s for s in cleaned if s.lower() not in {"nan", "none", "sheet1"}]
    if not cleaned:
        return []
    # Let the LLM pick the actual companies out of the labels (row names like "Revenue" are not companies)
    try:
        found = classify_companies("\n".join(dict.fromkeys(cleaned)))
    except Exception as e:
        print(f"[company_normalizer] Excel company detection failed: {e}")
        return []
    return canonicalize(found)
