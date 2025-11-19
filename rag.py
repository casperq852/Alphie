from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import faiss
from pypdf import PdfReader
import tiktoken
import sqlite3, re, os, json, hashlib, math
import pandas as pd
from company_normalizer import _extract_company_names_from_excel, load_company_index

from dateutil.parser import parse as dtparse
import dateparser

from openai import OpenAI
from db import get_conn, DATA_DIR

# -------- Models --------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # default answering model
MINI_MODEL   = os.getenv("MINI_MODEL", "gpt-4o-mini")    # forced for ingestion steps
EMB_MODEL    = "text-embedding-3-large"
client = OpenAI()

USE_QDRANT = bool(os.getenv("USE_QDRANT", "").strip()) or bool(os.getenv("QDRANT_URL", "").strip())
if USE_QDRANT:
    from qdrant_backend import upsert_chunks as qdrant_upsert, search as qdrant_search

DETAIL_PRESETS = {
    "brief":      {"k": 16, "per_doc_cap": 3, "max_tokens": 900},
    "normal":     {"k": 24, "per_doc_cap": 4, "max_tokens": 1400},
    "elaborate":  {"k": 36, "per_doc_cap": 6, "max_tokens": 2200},
    "exhaustive": {"k": 48, "per_doc_cap": 8, "max_tokens": 3000},

    # Research mode (bigger budgets)
    "deep_dive":  {"k": 64, "per_doc_cap": 10, "max_tokens": 6000},
}

# ---------- On-disk stores ----------
VS_DIR = (DATA_DIR / "vector_store"); VS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = (DATA_DIR / "docs");        DOCS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = (DATA_DIR / "models");    MODELS_DIR.mkdir(parents=True, exist_ok=True)  # NEW
INDEX_PATH = (VS_DIR / "index.faiss")
META_PATH  = (VS_DIR / "meta.json")

# ---------- Date helpers ----------
_MONTHS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)"
DATE_PATTERNS = [
    r"\b\d{4}-\d{1,2}-\d{1,2}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{1,2}\s+" + _MONTHS + r"\s+\d{4}\b",
    r"\b" + _MONTHS + r"\s+\d{1,2},\s*\d{4}\b",
]

def list_all_companies() -> List[str]:
    """
    Return all companies stored in the DB from document ingestion.
    Used to build embedding index for canonicalization.
    """
    conn = get_conn()
    rows = conn.execute("SELECT companies FROM documents").fetchall()
    conn.close()

    out = set()
    for r in rows:
        try:
            lst = json.loads(r[0]) if r[0] else []
            for c in lst:
                if c.strip():
                    out.add(c.strip())
        except Exception:
            pass

    return sorted(out)

load_company_index(list_all_companies())

def _safe_iso(d: Optional[datetime]) -> Optional[str]:
    return d.date().isoformat() if d else None

def _date_from_text(text: str, prefer_recent: bool = True) -> Optional[str]:
    if not text: return None
    cands = set()
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            cands.add(m.group(0))
    parsed: List[datetime] = []
    for c in cands:
        d = dateparser.parse(c)
        if d: parsed.append(d.replace(tzinfo=None))
    parsed = [p for p in parsed if p <= datetime.now()]
    if not parsed: return None
    best = max(parsed) if prefer_recent else min(parsed)
    return _safe_iso(best)

def _dates_in_text(text: str) -> List[str]:
    out = []
    if not text: return out
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            d = dateparser.parse(m.group(0))
            if d: out.append(d.date().isoformat())
    return sorted(set(out), reverse=True)

def _pdf_metadata_to_json(meta_obj) -> Dict[str, Any]:
    try:
        d = dict(meta_obj or {})
        return {str(k): (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in d.items()}
    except Exception:
        try:
            return json.loads(str(meta_obj)) if meta_obj else {}
        except Exception:
            return {}

def _maybe_parse_pdf_meta_date(v) -> Optional[datetime]:
    if not v: return None
    s = str(v)
    m = re.match(r"^D:(\d{4})(\d{2})?(\d{2})?", s)
    if m:
        y = int(m.group(1)); mo = int(m.group(2) or 1); da = int(m.group(3) or 1)
        try: return datetime(y, mo, da)
        except Exception: return None
    try:
        return dtparse(s, fuzzy=True).replace(tzinfo=None)
    except Exception:
        return None

def _extract_doc_date_from_metadata(meta_obj) -> Optional[str]:
    keys = ["/CreationDate", "/ModDate", "CreationDate", "ModDate", "creation_date", "mod_date"]
    try:
        for k in keys:
            if hasattr(meta_obj, "get") and meta_obj.get(k):
                iso = _safe_iso(_maybe_parse_pdf_meta_date(meta_obj.get(k)))
                if iso: return iso
    except Exception:
        pass
    try:
        for k in keys:
            v = getattr(meta_obj, k, None)
            iso = _safe_iso(_maybe_parse_pdf_meta_date(v))
            if iso: return iso
    except Exception:
        pass
    try:
        return _date_from_text(str(meta_obj))
    except Exception:
        return None

def _parse_query_date_focus(q: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    q_low = q.lower().strip()
    now = datetime.now()
    rel = {
        "today": (now.date(), now.date()),
        "yesterday": ((now - timedelta(days=1)).date(), (now - timedelta(days=1)).date()),
        "last week": ((now - timedelta(days=7)).date(), now.date()),
        "past week": ((now - timedelta(days=7)).date(), now.date()),
        "past month": ((now - timedelta(days=30)).date(), now.date()),
        "last month": ((now - timedelta(days=30)).date(), now.date()),
    }
    for k, (a, b) in rel.items():
        if k in q_low:
            a_iso, b_iso = a.isoformat(), b.isoformat()
            return (a_iso, None, None) if a_iso == b_iso else (None, a_iso, b_iso)
    dates = _dates_in_text(q)
    if len(dates) == 1: return dates[0], None, None
    if len(dates) >= 2:
        d_sorted = sorted(dates)
        return None, d_sorted[0], d_sorted[-1]
    return None, None, None

# ---------- Index I/O ----------
def _load_index_and_meta():
    VS_DIR.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatIP(3072)
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    else:
        meta = {"ids": [], "map": {}}
    return index, meta

def _save_index_and_meta(index, meta):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta), encoding="utf-8")

def pdf_to_text(path: Path) -> tuple[str, int, Dict[str, Any]]:
    r = PdfReader(str(path))
    texts = [(p.extract_text() or "") for p in r.pages]
    return "\n".join(texts), len(r.pages), {"metadata": _pdf_metadata_to_json(r.metadata)}

def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def embed_texts(texts: List[str]):
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs

def chunk_text(text: str, target_tokens: int = 500, overlap_tokens: int = 50) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text or "")
    chunks, i = [], 0
    while i < len(ids):
        chunks.append(enc.decode(ids[i:i+target_tokens]))
        i += max(1, target_tokens - overlap_tokens)
    return [c for c in chunks if c.strip()]

# ---- filters (drug names + brokers/data vendors) ----
_DRUG_SUFFIXES = (
    "mab","zumab","ximab","limab","tinib","nib","ciclib","parib","setron","mycin","floxacin",
    "dazole","oxetine","pril","sartan","statin","gliptin","gliflozin","prazole","caine",
    "dronate","avir","rudin","arin","amine","tamide","terol","dopa","prost","sal","afil",
)
_BAD_TOKENS = {"tablet","solution","capsule","injection","mg","ml","iv","po","qd","bid","tid","q4w"}
_BROKER_VENDORS = {
    "ubs","morgan stanley","goldman sachs","jpmorgan","bank of america","bofa",
    "citigroup","citi","credit suisse","barclays","deutsche bank","societe generale",
    "société générale","bnp paribas","kepler cheuvreux","berenberg","bernstein",
    "cowen","td cowen","rbc","jefferies","wells fargo","evercore isi","piper sandler",
    "oppenheimer","raymond james","stifel","cantor fitzgerald","roth mkm","redburn",
    "oddo","oddo bhf","commerzbank","macquarie","mizuho","nomura","daiwa","smbc",
    "mufg","standard chartered","ing","abn amro","seb","nordea","danske bank",
    "swedbank","intesa sanpaolo","mediobanca","caixabank","kbw","keefe bruyette",
    "numis","peel hunt","liberum","investec","arete","wolfe","wedbush","truist",
    "keybanc","needham","susquehanna","rosenblatt","guggenheim","exane","exane bnp",
    "hsbc","factset","bloomberg","refinitiv","morningstar","s&p global","spglobal",
    "lseg","london stock exchange group"
}

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

def _clean_company_list(items: List[str]) -> List[str]:
    clean, seen = [], set()
    for x in items:
        s = re.sub(r"\s+", " ", str(x)).strip()
        if not s or len(s) < 2: continue
        low = s.lower()
        if any(tok in low for tok in _BAD_TOKENS): continue
        if any(low.endswith(sfx) for sfx in _DRUG_SUFFIXES): continue
        if any(b in low for b in _BROKER_VENDORS): continue
        if re.search(r"[A-Za-z]\d{2,}", s) and "-" in s: continue
        if not re.search(r"[A-Za-z]", s): continue
        if s.islower(): continue
        key = s.lower()
        if key in seen: continue
        seen.add(key); clean.append(s)
    return clean[:8]

def classify_companies(raw_text: str, model: Optional[str] = None) -> List[str]:
    """
    Extract canonical company names or tickers from raw_text.
    'model' lets the caller choose (UI-selected for chat; MINI for ingestion).
    """
    system = (
        "Extract public company tickers or canonical company names mentioned in a broker research PDF or user query. "
        "EXCLUDE drug names, molecules, trial arms, medicine brands, dosages, medical terms, and "
        "publishers/brokers/data vendors (e.g., UBS, HSBC, Morgan Stanley, Goldman Sachs, JPMorgan, etc.). "
        "Return ONLY a JSON array of strings (no code fences). Return all the companies found, also if the name of the company is mentioned differently multiple times. Return each of those seperately as well as tickers if they are found."
    )
    user = f"Text:\n{raw_text[:8000]}"
    use_model = "gpt-4o-mini"
    resp = client.chat.completions.create(
        model=use_model, temperature=0.2,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}]
    )
    content = _strip_code_fences(resp.choices[0].message.content.strip())
    try:
        data = json.loads(content)
        items = [str(x) for x in data] if isinstance(data, list) else []
    except Exception:
        items = [p for p in re.split(r"[,;\n]+", content) if p.strip()]
    return _clean_company_list(items)

# -------- ingestion / search --------
def add_documents(paths: List[Path]) -> Dict[str, Any]:
    conn = get_conn()
    index, meta = _load_index_and_meta()
    added = skipped = 0
    details = []
    for p in paths:
        if p.suffix.lower() != ".pdf":
            details.append({"file": p.name, "status": "ignored", "reason": "not a PDF"}); continue
        try:
            raw = p.read_bytes()
        except Exception as e:
            details.append({"file": p.name, "status": "error", "error": f"read_bytes: {e}"}); continue
        h = hash_bytes(raw)
        try:
            row = conn.execute("SELECT id FROM documents WHERE hash=?", (h,)).fetchone()
            if row:
                skipped += 1
                details.append({"file": p.name, "status": "skipped_duplicate", "hash": h})
                continue

            safe_name = f"{h[:10]}_{p.name}"
            stored_path = (DOCS_DIR / safe_name); stored_path.write_bytes(raw)

            text, n_pages, extras = pdf_to_text(stored_path)
            metadata_json = extras.get("metadata", {})
            meta_date = _extract_doc_date_from_metadata(metadata_json) if metadata_json else None
            first_pages_text = "\n".join((text or "").split("\n")[:1200])
            text_date = _date_from_text(first_pages_text) or _date_from_text(text)
            doc_date = meta_date or text_date

            # ALWAYS use MINI for ingestion-time LLM work
            companies = classify_companies(text, model=MINI_MODEL)

            conn.execute(
                "INSERT INTO documents(filename, hash, companies, n_pages, stored_path, doc_date, meta_json) "
                "VALUES(?,?,?,?,?,?,?)",
                (p.name, h, json.dumps(companies), n_pages, str(stored_path), doc_date, json.dumps(metadata_json)),
            )
            doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            chunks = chunk_text(text)
            if not chunks:
                conn.execute("DELETE FROM documents WHERE id=?", (doc_id,)); conn.commit()
                details.append({"file": p.name, "status": "error", "error": "no chunks produced"}); continue

            vecs = embed_texts(chunks)

            if USE_QDRANT:
                qdrant_upsert(
                    doc_id=doc_id,
                    vectors=vecs.tolist(),
                    chunks=chunks,
                    companies_json=json.dumps(companies),
                    stored_path=str(stored_path),
                    doc_date=doc_date,
                )
                for i, chunk in enumerate(chunks):
                    chunk_dates = _dates_in_text(chunk)
                    chunk_date = chunk_dates[0] if chunk_dates else doc_date
                    conn.execute(
                        "INSERT INTO chunks(document_id, company, chunk_index, text, embedding, chunk_date) VALUES(?,?,?,?,?,?)",
                        (doc_id, json.dumps(companies), i, chunk, b"", chunk_date),
                    )
            else:
                start_id = len(meta["ids"])
                ids = list(range(start_id, start_id + len(chunks)))
                index.add(vecs)
                for i, (cid, chunk) in enumerate(zip(ids, chunks)):
                    meta["ids"].append(cid)
                    meta["map"][str(cid)] = {"document_id": doc_id, "chunk_index": i}
                    chunk_dates = _dates_in_text(chunk)
                    chunk_date = chunk_dates[0] if chunk_dates else doc_date
                    conn.execute(
                        "INSERT INTO chunks(document_id, company, chunk_index, text, embedding, chunk_date) VALUES(?,?,?,?,?,?)",
                        (doc_id, json.dumps(companies), i, chunk, vecs[i].tobytes(), chunk_date),
                    )

            conn.commit()
            details.append({
                "file": p.name, "status": "added", "hash": h,
                "companies": companies, "pages": n_pages, "chunks": len(chunks),
                "stored_path": str(stored_path), "doc_date": doc_date
            })
            added += 1

        except sqlite3.IntegrityError as e:
            skipped += 1; details.append({"file": p.name, "status": "skipped_duplicate", "hash": h, "note": str(e)}); conn.rollback()
        except Exception as e:
            conn.rollback(); details.append({"file": p.name, "status": "error", "error": str(e)})
    _save_index_and_meta(index, meta); conn.close()
    return {"added": added, "skipped": skipped, "details": details}

def search_chunks(
    query: str,
    k: int = 12,
    company_filter: Optional[List[str]] = None
):
    if USE_QDRANT:
        qvec = embed_texts([query])[0].tolist()
        exact_date, date_from, date_to = _parse_query_date_focus(query)
        hits = qdrant_search(
            query_vec=qvec,
            k=k,
            company_filter=company_filter or [],
            date_focus=(exact_date, date_from, date_to),
        )
        conn = get_conn()
        out = []
        for h in hits:
            row = conn.execute(
                "SELECT filename FROM documents WHERE id=? LIMIT 1", (h["document_id"],)
            ).fetchone()
            source = row[0] if row else ""
            h["source"] = source
            out.append(h)
        conn.close()
        return out

    index, meta = _load_index_and_meta()
    if not meta["ids"]:
        return []
    qvec = embed_texts([query])
    D, I = index.search(qvec, min(k * 4, len(meta["ids"])))
    conn = get_conn(); candidates = []

    def _norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"\b(group|plc|inc|nv|sa|ltd|corp|co|ag|se|nv)\b", " ", s)
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    cf_norm = [_norm(c) for c in (company_filter or [])]
    exact_date, date_from, date_to = _parse_query_date_focus(query)
    focus_from = date_from or exact_date
    focus_to = date_to or exact_date

    for cid, score in zip(I[0], D[0]):
        if cid < 0: continue
        m = meta["map"].get(str(cid), {})
        if not m: continue
        row = conn.execute(
            """SELECT c.text, c.chunk_date, d.filename, d.id, d.companies, d.stored_path, d.added_at, d.doc_date
               FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE c.document_id=? AND c.chunk_index=? LIMIT 1""",
            (m["document_id"], m["chunk_index"]),
        ).fetchone()
        if not row: continue
        text, chunk_date, filename, doc_id, companies_json, stored_path, added_at, doc_date = row
        try: comps = json.loads(companies_json or "[]")
        except Exception: comps = []

        if cf_norm:
            comp_norms = [_norm(c) for c in comps]
            ok = any(any(f in c for c in comp_norms) for f in cf_norm)
            if not ok:
                continue

        adj = float(score)

        base_date = None
        for d in (chunk_date, doc_date, added_at):
            if d:
                try:
                    base_date = datetime.fromisoformat(str(d))
                    break
                except Exception:
                    pass
        age_days = None
        if base_date:
            age_days = max(0, (datetime.now() - base_date).days)
            adj *= math.exp(- (age_days / 120.0))

        if exact_date or (focus_from and focus_to):
            try:
                cd = datetime.fromisoformat(chunk_date) if chunk_date else None
            except Exception:
                cd = None
            if exact_date:
                if cd:
                    delta = abs((cd.date() - datetime.fromisoformat(exact_date).date()).days)
                    adj *= math.exp(- (delta ** 2) / (2 * (2.0 ** 2))) * 1.5
                else:
                    adj *= 0.85
            else:
                try:
                    f = datetime.fromisoformat(focus_from).date()
                    t = datetime.fromisoformat(focus_to).date()
                    if cd and f <= cd.date() <= t:
                        adj *= 1.4
                    elif age_days is not None:
                        adj *= 0.9
                except Exception:
                    pass

        candidates.append({
            "score": adj, "text": text, "companies": comps,
            "document_id": doc_id, "chunk_index": m["chunk_index"],
            "source": filename, "stored_path": stored_path,
            "chunk_date": chunk_date, "doc_date": doc_date
        })

    conn.close()
    candidates.sort(key=lambda x: x["score"], reverse=True)
    # if cf_norm and len(candidates) < max(2, k // 3):
    #     return search_chunks(query, k=k, company_filter=None)
    return candidates[:k]

def collect_evidence(query: str, k: int, per_doc_cap: int, company_filter=None):
    hits = search_chunks(query, k=k, company_filter=company_filter)
    by_doc = {}
    for h in hits:
        key = (h.get("document_id"), h.get("source"))
        by_doc.setdefault(key, []).append(h)
    pooled = []
    for key, lst in by_doc.items():
        lst.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        cap = max(2, per_doc_cap - 1)
        pooled.extend(lst[:cap])
    pooled.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return pooled[:k]


# Expose helper to UI (imported in app.py)
def _parse_query_date_focus_ui(q: str):
    return _parse_query_date_focus(q)

# Keep name for app import
_parse_query_date_focus = _parse_query_date_focus

# ---- Metric canonicalization (for Excel models) ----

_CANONICAL_METRICS = [
    "revenue",
    "gross_profit",
    "ebitda",
    "ebit",
    "operating_income",
    "operating_margin",
    "gross_margin",
    "net_income",
    "eps",
    "adj_eps",
    "free_cash_flow",
    "operating_cash_flow",
    "capex",
    "working_capital",
    "total_debt",
    "net_debt",
    "equity",
    "assets",
    "roic",
    "roe",
    "roa",
    "fcf_yield",
    "dividend",
    "dividend_per_share",
    "dividend_yield",
    "shares_outstanding",
    "net_margin",
    "sg_and_a",
    "r_and_d",
    "cost_of_goods_sold",
    "interest_expense",
    "tax_expense",
    "other",
]

def _canonicalize_metric_name(raw_name: str) -> tuple[str, bool]:
    """
    Map a raw metric row name from an Excel model to a canonical metric name
    and a flag whether it's a ratio (margin, yield, etc.).

    Returns: (canonical_name, is_ratio)
    """
    if not raw_name or not str(raw_name).strip():
        return "other", False

    raw = str(raw_name).strip()

    # quick heuristics to save LLM calls on obvious stuff
    low = raw.lower()
    if "margin" in low:
        return "operating_margin" if "operat" in low else "gross_margin", True
    if "eps" in low:
        return "eps", False
    if "revenue" in low or "sales" in low or "turnover" in low:
        return "revenue", False
    if "ebitda" in low:
        return "ebitda", False
    if "ebit" in low and "ebitda" not in low:
        return "ebit", False
    if "free cash flow" in low or "fcf" in low:
        return "free_cash_flow", False
    if "net income" in low or ("net profit" in low):
        return "net_income", False

    system = (
        "You map raw financial metric names from Excel models to a compact canonical taxonomy. "
        "Pick the closest match from this list: "
        + ", ".join(_CANONICAL_METRICS)
        + ". "
        "Also decide if the metric is a ratio (margin, yield, %, ROIC, ROE etc.). "
        "Respond ONLY with JSON: {\"canonical\": \"...\", \"is_ratio\": true/false}."
    )
    user = f'Raw metric name: "{raw}"'

    try:
        resp = client.chat.completions.create(
            model=MINI_MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = resp.choices[0].message.content.strip()
        content = _strip_code_fences(content)
        data = json.loads(content)
        canonical = str(data.get("canonical", "other")).strip().lower()
        if canonical not in _CANONICAL_METRICS:
            canonical = "other"
        is_ratio = bool(data.get("is_ratio", False))
        return canonical, is_ratio
    except Exception:
        return "other", False

def _extract_metrics_from_excel(path: Path) -> list[dict]:
    """
    Very pragmatic parser:
    - Assumes first column = metric name
    - Other columns = periods (e.g. 2022, 2023E, FY24E)
    - Stores numeric cells as (metric, period, value)
    No currency / unit detection yet -> leave as None for now.
    """
    metrics: list[dict] = []
    try:
        xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        return [{"error": f"read_excel failed: {e}"}]

    # collect all unique row names up front for LLM canonicalization
    raw_names_set = set()
    for sheet_name, df in xls.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        # assume first column = row labels
        first_col = df.columns[0]
        for raw_name in df[first_col].tolist():
            if isinstance(raw_name, str) and raw_name.strip():
                raw_names_set.add(raw_name.strip())

    canonical_map: dict[str, tuple[str, bool]] = {}
    for rn in raw_names_set:
        canonical_map[rn] = _canonicalize_metric_name(rn)

    for sheet_name, df in xls.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        first_col = df.columns[0]

        for _, row in df.iterrows():
            raw_name = row.get(first_col)
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            raw_name = raw_name.strip()
            canonical_name, is_ratio = canonical_map.get(raw_name, ("other", False))

            for col in df.columns[1:]:
                period = str(col).strip()
                val = row.get(col)
                # keep only numeric cells
                try:
                    if pd.isna(val):
                        continue
                except Exception:
                    pass
                if not isinstance(val, (int, float, np.number)):
                    continue

                metrics.append(
                    {
                        "raw_name": raw_name,
                        "canonical_name": canonical_name,
                        "period": period,
                        "value": float(val),
                        "currency": None,
                        "unit": None,
                        "is_ratio": is_ratio,
                    }
                )
    return metrics

def add_models(paths: List[Path]) -> Dict[str, Any]:
    """
    Ingest Excel analyst models (.xlsx):
    - hash & store file under DATA_DIR/models
    - detect companies (roughly) using classify_companies on filename
    - parse metrics with _extract_metrics_from_excel
    - write to models + metrics tables
    """
    conn = get_conn()
    added = skipped = 0
    details: list[dict] = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for p in paths:
        suffix = p.suffix.lower()
        if suffix not in (".xlsx", ".xlsm"):
            details.append({"file": p.name, "status": "ignored", "reason": "not an Excel model"})
            continue

        try:
            raw_bytes = p.read_bytes()
        except Exception as e:
            details.append({"file": p.name, "status": "error", "error": f"read_bytes: {e}"})
            continue

        h = hash_bytes(raw_bytes)

        try:
            row = conn.execute("SELECT id FROM models WHERE hash=?", (h,)).fetchone()
            if row:
                skipped += 1
                details.append({"file": p.name, "status": "skipped_duplicate", "hash": h})
                continue

            safe_name = f"{h[:10]}_{p.name}"
            stored_path = MODELS_DIR / safe_name
            stored_path.write_bytes(raw_bytes)

            # crude company detection from filename; can be improved later
            # 1. Try extraction from Excel contents
            companies = _extract_company_names_from_excel(stored_path)

            # 2. If still empty, fallback to filename classifier
            if not companies:
                try:
                    companies = classify_companies(p.name, model=MINI_MODEL)
                except Exception:
                    companies = []

            conn.execute(
                "INSERT INTO models(filename, hash, companies, stored_path) VALUES(?,?,?,?)",
                (p.name, h, json.dumps(companies), str(stored_path)),
            )
            model_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            parsed_metrics = _extract_metrics_from_excel(stored_path)

            n_metrics = 0
            for m in parsed_metrics:
                if "error" in m:
                    # bubble up parsing error but keep model row
                    details.append(
                        {
                            "file": p.name,
                            "status": "parse_warning",
                            "error": m["error"],
                        }
                    )
                    continue

                raw_name = m["raw_name"]
                canonical_name = m["canonical_name"]
                period = m["period"]
                value = m["value"]
                currency = m["currency"]
                unit = m["unit"]
                is_ratio = 1 if m["is_ratio"] else 0

                # if multiple companies detected, duplicate metric per company; otherwise store with empty company
                if companies:
                    for c in companies:
                        conn.execute(
                            """
                            INSERT INTO metrics(
                                model_id, company, raw_name, canonical_name,
                                period, value, currency, unit, is_ratio
                            ) VALUES(?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                model_id,
                                c,
                                raw_name,
                                canonical_name,
                                period,
                                value,
                                currency,
                                unit,
                                is_ratio,
                            ),
                        )
                        n_metrics += 1
                else:
                    conn.execute(
                        """
                        INSERT INTO metrics(
                            model_id, company, raw_name, canonical_name,
                            period, value, currency, unit, is_ratio
                        ) VALUES(?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            model_id,
                            "",
                            raw_name,
                            canonical_name,
                            period,
                            value,
                            currency,
                            unit,
                            is_ratio,
                        ),
                    )
                    n_metrics += 1

            conn.commit()
            added += 1
            details.append(
                {
                    "file": p.name,
                    "status": "added",
                    "hash": h,
                    "companies": companies,
                    "n_metrics": n_metrics,
                    "stored_path": str(stored_path),
                }
            )

        except sqlite3.IntegrityError as e:
            skipped += 1
            details.append(
                {"file": p.name, "status": "skipped_duplicate", "hash": h, "note": str(e)}
            )
            conn.rollback()
        except Exception as e:
            conn.rollback()
            details.append({"file": p.name, "status": "error", "error": str(e)})

    conn.close()
    return {"added": added, "skipped": skipped, "details": details}

