"""Ingestion (PDF research + Excel models) and retrieval over the local research repository."""

import hashlib
import json
import os
import re
import sqlite3
import threading
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import tiktoken
from dateutil.parser import parse as dtparse
from pypdf import PdfReader

from company_normalizer import (
    _extract_company_names_from_excel,
    canonicalize,
    classify_companies,
    load_company_index,
)
from db import DATA_DIR, get_conn
from llm import EMB_DIM, INGEST_MODEL, chat, embed, strip_code_fences
from scoring import (
    adjust_score,
    date_from_text,
    dates_in_text,
    names_match,
    parse_query_date_focus,
    to_date,
)

__all__ = [
    "DETAIL_PRESETS", "add_documents", "add_models", "canonicalize", "classify_companies", "collect_evidence",
    "delete_document", "list_all_companies", "rebuild_faiss_index", "refresh_company_index", "search_chunks",
]  # fmt: skip

# Ingestion performance/safety knobs
INGEST_MAX_CHUNKS = int(os.getenv("INGEST_MAX_CHUNKS", "160"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
_EXACT_SEARCH_BATCH = 2000  # rows per batch when scanning stored embeddings for filtered search


def _env_flag(name: str) -> bool | None:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return None
    return v in {"1", "true", "yes", "on"}


# Explicit USE_QDRANT wins (so USE_QDRANT=false really disables it); otherwise legacy rule: QDRANT_URL set -> on.
_flag = _env_flag("USE_QDRANT")
USE_QDRANT = _flag if _flag is not None else bool(os.getenv("QDRANT_URL", "").strip())
if USE_QDRANT:
    import qdrant_backend  # noqa: PLC0415  (optional dependency, only when enabled)

DETAIL_PRESETS = {
    "brief": {"k": 16, "per_doc_cap": 3, "max_tokens": 900},
    "normal": {"k": 24, "per_doc_cap": 4, "max_tokens": 1400},
    "elaborate": {"k": 36, "per_doc_cap": 6, "max_tokens": 2200},
    "exhaustive": {"k": 48, "per_doc_cap": 8, "max_tokens": 3000},
    "deep_dive": {"k": 64, "per_doc_cap": 10, "max_tokens": 6000},
}

# ---------- On-disk stores ----------
VS_DIR = DATA_DIR / "vector_store"
DOCS_DIR = DATA_DIR / "docs"
MODELS_DIR = DATA_DIR / "models"
for _d in (VS_DIR, DOCS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
INDEX_PATH = VS_DIR / "index.faiss"
META_PATH = VS_DIR / "meta.json"

# One process serves all Streamlit sessions: serialise writers so two uploads cannot corrupt the index.
_INGEST_LOCK = threading.RLock()
_index_cache: dict[str, Any] = {"mtime": None, "index": None, "meta": None}


# ---------- Company registry ----------
def list_all_companies() -> list[str]:
    """All companies tagged on ingested documents (used for canonicalisation)."""
    conn = get_conn()
    rows = conn.execute("SELECT companies FROM documents").fetchall()
    conn.close()
    out = set()
    for (raw,) in rows:
        try:
            out.update(c.strip() for c in json.loads(raw or "[]") if c and c.strip())
        except Exception:
            pass
    return sorted(out)


def refresh_company_index() -> None:
    """Cheap when nothing changed (name embeddings are cached on disk)."""
    load_company_index(list_all_companies())


# ---------- PDF helpers ----------
def _pdf_metadata_to_json(meta_obj) -> dict[str, Any]:
    try:
        d = dict(meta_obj or {})
        return {str(k): (v if isinstance(v, (int, float, bool)) else str(v)) for k, v in d.items()}
    except Exception:
        return {}


def _maybe_parse_pdf_meta_date(v) -> datetime | None:
    if not v:
        return None
    s = str(v)
    if s.startswith("D:") and len(s) >= 6:
        try:
            y, mo, da = int(s[2:6]), int(s[6:8] or 1), int(s[8:10] or 1)
            return datetime(y, mo, da)
        except Exception:
            return None
    try:
        return dtparse(s, fuzzy=True).replace(tzinfo=None)
    except Exception:
        return None


def _extract_doc_date_from_metadata(meta: dict[str, Any]) -> str | None:
    for k in ("/CreationDate", "CreationDate", "/ModDate", "ModDate"):
        d = _maybe_parse_pdf_meta_date(meta.get(k))
        if d and d.date() <= date.today():
            return d.date().isoformat()
    return None


def pdf_to_text(path: Path) -> tuple[list[str], dict[str, Any]]:
    r = PdfReader(str(path))
    pages = []
    for p in r.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return pages, _pdf_metadata_to_json(r.metadata)


def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@lru_cache(maxsize=1)
def _encoder():
    return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, target_tokens: int = CHUNK_TOKENS, overlap_tokens: int = CHUNK_OVERLAP) -> list[str]:
    enc = _encoder()
    ids = enc.encode(text or "")
    step = max(1, target_tokens - overlap_tokens)
    chunks = [enc.decode(ids[i : i + target_tokens]) for i in range(0, len(ids), step)]
    return [c for c in chunks if c.strip()]


# ---------- FAISS index I/O ----------
def _load_index_and_meta():
    """Cached in memory; reloaded only when the file on disk changed."""
    mtime = INDEX_PATH.stat().st_mtime if INDEX_PATH.exists() else None
    if _index_cache["index"] is not None and _index_cache["mtime"] == mtime:
        return _index_cache["index"], _index_cache["meta"]
    index = faiss.read_index(str(INDEX_PATH)) if INDEX_PATH.exists() else faiss.IndexFlatIP(EMB_DIM)
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {"ids": [], "map": {}}
    _index_cache.update(mtime=mtime, index=index, meta=meta)
    return index, meta


def _save_index_and_meta(index, meta) -> None:
    tmp_idx, tmp_meta = INDEX_PATH.with_suffix(".tmp"), META_PATH.with_suffix(".tmp")
    faiss.write_index(index, str(tmp_idx))
    tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
    tmp_meta.replace(META_PATH)
    tmp_idx.replace(INDEX_PATH)
    _index_cache.update(mtime=INDEX_PATH.stat().st_mtime, index=index, meta=meta)


def rebuild_faiss_index() -> int:
    """Rebuild the FAISS index from embeddings stored in SQLite (repair / after deletes)."""
    with _INGEST_LOCK:
        index, meta = faiss.IndexFlatIP(EMB_DIM), {"ids": [], "map": {}}
        conn = get_conn()
        cur = conn.execute(
            "SELECT document_id, chunk_index, embedding FROM chunks WHERE length(embedding) > 0 ORDER BY id"
        )
        while rows := cur.fetchmany(_EXACT_SEARCH_BATCH):
            vecs = np.vstack([np.frombuffer(r[2], dtype="float32") for r in rows])
            start = len(meta["ids"])
            index.add(vecs)
            for j, (doc_id, chunk_index, _) in enumerate(rows):
                meta["ids"].append(start + j)
                meta["map"][str(start + j)] = {"document_id": doc_id, "chunk_index": chunk_index}
        conn.close()
        _save_index_and_meta(index, meta)
        return len(meta["ids"])


# ---------- Ingestion ----------
def _insert_chunks(conn, doc_id: int, chunks: list[str], vecs: np.ndarray, companies: list[str], dates: list):
    blob_vecs = not USE_QDRANT  # Qdrant keeps the vectors itself
    conn.executemany(
        "INSERT INTO chunks(document_id, company, chunk_index, text, embedding, chunk_date) VALUES(?,?,?,?,?,?)",
        [
            (doc_id, json.dumps(companies), i, c, vecs[i].tobytes() if blob_vecs else b"", dates[i])
            for i, c in enumerate(chunks)
        ],
    )


def _ingest_pdf(conn, p: Path) -> dict[str, Any]:
    raw = p.read_bytes()
    h = hash_bytes(raw)
    if conn.execute("SELECT id FROM documents WHERE hash=?", (h,)).fetchone():
        return {"file": p.name, "status": "skipped_duplicate", "hash": h}

    stored_path = DOCS_DIR / f"{h[:10]}_{p.name}"
    stored_path.write_bytes(raw)
    pages, metadata = pdf_to_text(stored_path)
    text = "\n".join(pages)
    if not text.strip():
        stored_path.unlink(missing_ok=True)
        return {"file": p.name, "status": "error", "error": "no extractable text (scanned PDF?)"}

    doc_date = _extract_doc_date_from_metadata(metadata) or date_from_text("\n".join(pages[:2])) or date_from_text(text)
    companies = classify_companies(text, model=INGEST_MODEL)

    chunks = chunk_text(text)
    total_chunks = len(chunks)
    if INGEST_MAX_CHUNKS > 0:
        chunks = chunks[:INGEST_MAX_CHUNKS]
    vecs = embed(chunks, batch_size=EMBED_BATCH_SIZE)
    chunk_dates = [(dates_in_text(c) or [doc_date])[0] for c in chunks]

    cur = conn.execute(
        "INSERT INTO documents(filename, hash, companies, n_pages, stored_path, doc_date, meta_json) "
        "VALUES(?,?,?,?,?,?,?)",
        (p.name, h, json.dumps(companies), len(pages), str(stored_path), doc_date, json.dumps(metadata)),
    )
    doc_id = cur.lastrowid
    _insert_chunks(conn, doc_id, chunks, vecs, companies, chunk_dates)

    if USE_QDRANT:
        qdrant_backend.delete_document(doc_id)  # clear leftovers of an earlier failed ingest that used this id
        qdrant_backend.upsert_chunks(
            doc_id=doc_id,
            vectors=vecs.tolist(),
            chunks=chunks,
            companies=companies,
            stored_path=str(stored_path),
            doc_date=doc_date,
            chunk_dates=chunk_dates,
        )
        conn.commit()
    else:
        # Commit the DB first, then extend the index: a failed insert can never leave vectors in FAISS that
        # point at a rolled-back (and later re-used) document id.
        conn.commit()
        index, meta = _load_index_and_meta()
        start = len(meta["ids"])
        index.add(vecs)
        for i in range(len(chunks)):
            meta["ids"].append(start + i)
            meta["map"][str(start + i)] = {"document_id": doc_id, "chunk_index": i}
        _save_index_and_meta(index, meta)

    return {
        "file": p.name, "status": "added", "hash": h, "companies": companies, "pages": len(pages),
        "chunks": len(chunks), "chunks_total": total_chunks, "truncated_for_ingest": total_chunks > len(chunks),
        "doc_date": doc_date,
    }  # fmt: skip


def add_documents(paths: list[Path]) -> dict[str, Any]:
    added = skipped = 0
    details = []
    with _INGEST_LOCK:
        conn = get_conn()
        for p in paths:
            if p.suffix.lower() != ".pdf":
                details.append({"file": p.name, "status": "ignored", "reason": "not a PDF"})
                continue
            try:
                res = _ingest_pdf(conn, p)
            except sqlite3.IntegrityError as e:
                conn.rollback()
                res = {"file": p.name, "status": "skipped_duplicate", "note": str(e)}
            except Exception as e:
                conn.rollback()
                res = {"file": p.name, "status": "error", "error": f"{type(e).__name__}: {e}"}
            added += res["status"] == "added"
            skipped += res["status"] == "skipped_duplicate"
            details.append(res)
        conn.close()
    if added:
        try:
            refresh_company_index()
        except Exception as e:
            print(f"[rag] company index refresh failed: {e}")
    return {"added": added, "skipped": skipped, "details": details}


def delete_document(doc_id: int) -> bool:
    """Remove a document, its chunks, its stored file and its vectors."""
    with _INGEST_LOCK:
        conn = get_conn()
        row = conn.execute("SELECT stored_path FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            conn.close()
            return False
        conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        conn.commit()
        conn.close()
        if row[0]:
            Path(row[0]).unlink(missing_ok=True)
        if USE_QDRANT:
            qdrant_backend.delete_document(doc_id)
        else:
            rebuild_faiss_index()
    return True


# ---------- Retrieval ----------
def _candidate_doc_ids(
    conn, company_filter: list[str] | None, date_from: str | None, date_to: str | None
) -> list[int] | None:
    """Documents allowed by company / publication-date constraints; None = no constraint."""
    if not company_filter and not date_from and not date_to:
        return None
    f, t = to_date(date_from), to_date(date_to)
    out = []
    for doc_id, companies_json, doc_date, added_at in conn.execute(
        "SELECT id, companies, doc_date, added_at FROM documents"
    ):
        if f or t:
            d = to_date(doc_date) or to_date(added_at)
            if not d or (f and d < f) or (t and d > t):
                continue
        if company_filter:
            try:
                comps = json.loads(companies_json or "[]")
            except Exception:
                comps = []
            if not any(names_match(cf, c) for cf in company_filter for c in comps):
                continue
        out.append(doc_id)
    return out


def _exact_search(conn, qvec: np.ndarray, doc_ids: list[int], limit: int) -> list[tuple[float, int, int]]:
    """Brute-force cosine search over the stored embeddings of the given documents only.

    Filtering *before* ranking gives full recall for a company, instead of filtering the global top-k
    (which returns nothing when a company has few documents in a large repository)."""
    best: list[tuple[float, int, int]] = []
    for i in range(0, len(doc_ids), 500):
        ids = doc_ids[i : i + 500]
        cur = conn.execute(
            f"SELECT document_id, chunk_index, embedding FROM chunks "
            f"WHERE document_id IN ({','.join('?' * len(ids))}) AND length(embedding) > 0",
            ids,
        )
        while rows := cur.fetchmany(_EXACT_SEARCH_BATCH):
            mat = np.vstack([np.frombuffer(r[2], dtype="float32") for r in rows])
            sims = mat @ qvec
            best.extend((float(s), r[0], r[1]) for s, r in zip(sims, rows, strict=True))
            if len(best) > limit * 4:
                best = sorted(best, reverse=True)[:limit]
    return sorted(best, reverse=True)[:limit]


def _hydrate(conn, doc_id: int, chunk_index: int) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT c.text, c.chunk_date, d.filename, d.companies, d.stored_path, d.added_at, d.doc_date
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE c.document_id=? AND c.chunk_index=? LIMIT 1""",
        (doc_id, chunk_index),
    ).fetchone()
    if not row:
        return None
    text, chunk_date, filename, companies_json, stored_path, added_at, doc_date = row
    try:
        comps = json.loads(companies_json or "[]")
    except Exception:
        comps = []
    return {
        "text": text, "companies": comps, "document_id": doc_id, "chunk_index": chunk_index, "source": filename,
        "stored_path": stored_path, "chunk_date": chunk_date, "doc_date": doc_date, "added_at": added_at,
    }  # fmt: skip


def search_chunks(
    query: str,
    k: int = 12,
    company_filter: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    recency_tau_days: float | None = None,
) -> list[dict[str, Any]]:
    """Semantic search with optional company and publication-date constraints.

    Scores are shaped by recency (e-folding `recency_tau_days`, default RECENCY_TAU_DAYS; 0 disables) and by
    any date the query itself mentions ("vorige week", "2025-03-01", ...)."""
    date_focus = parse_query_date_focus(query)
    qvec = embed([query])[0]
    pool = max(k * 4, 40)
    conn = get_conn()
    try:
        doc_ids = _candidate_doc_ids(conn, company_filter, date_from, date_to)
        if doc_ids is not None and not doc_ids:
            return []

        candidates: list[dict[str, Any]] = []
        if USE_QDRANT:
            for h in qdrant_backend.search(qvec.tolist(), limit=pool, doc_ids=doc_ids):
                row = conn.execute("SELECT filename, added_at FROM documents WHERE id=?", (h["document_id"],))
                fetched = row.fetchone()
                if not fetched:
                    continue
                h["source"], h["added_at"] = fetched
                candidates.append(h)
        else:
            if doc_ids is not None:
                raw = _exact_search(conn, qvec, doc_ids, pool)
            else:
                index, meta = _load_index_and_meta()
                if not meta["ids"]:
                    return []
                scores, ids = index.search(qvec.reshape(1, -1), min(pool, len(meta["ids"])))
                raw = []
                for cid, score in zip(ids[0], scores[0], strict=True):
                    m = meta["map"].get(str(int(cid))) if cid >= 0 else None
                    if m:
                        raw.append((float(score), m["document_id"], m["chunk_index"]))
            for score, doc_id, chunk_index in raw:
                hit = _hydrate(conn, doc_id, chunk_index)
                if hit:
                    hit["score"] = score
                    candidates.append(hit)
    finally:
        conn.close()

    for h in candidates:
        h["similarity"] = h["score"]
        h["score"] = adjust_score(
            h["score"], h.get("chunk_date"), h.get("doc_date"), h.get("added_at"), date_focus, recency_tau_days
        )
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:k]


def collect_evidence(query: str, k: int, per_doc_cap: int, company_filter=None, **search_kwargs):
    """search_chunks + a cap on chunks per document, so one long report cannot crowd out the rest."""
    hits = search_chunks(query, k=k * 2, company_filter=company_filter, **search_kwargs)
    per_doc: dict[Any, int] = {}
    pooled = []
    for h in hits:
        key = h.get("document_id")
        if per_doc.get(key, 0) >= max(1, per_doc_cap):
            continue
        per_doc[key] = per_doc.get(key, 0) + 1
        pooled.append(h)
    return pooled[:k]


def days_ago_iso(days: int) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


# ---- Metric canonicalization (for Excel models) ----
_CANONICAL_METRICS = [
    "revenue", "gross_profit", "ebitda", "ebit", "operating_income", "operating_margin", "gross_margin",
    "net_income", "eps", "adj_eps", "free_cash_flow", "operating_cash_flow", "capex", "working_capital",
    "total_debt", "net_debt", "equity", "assets", "roic", "roe", "roa", "fcf_yield", "dividend",
    "dividend_per_share", "dividend_yield", "shares_outstanding", "net_margin", "ebitda_margin", "sg_and_a",
    "r_and_d", "cost_of_goods_sold", "interest_expense", "tax_expense", "other",
]  # fmt: skip


def _heuristic_metric(raw: str) -> tuple[str, bool] | None:
    low = raw.lower()
    if "/" in low or " to " in low:  # ratios like "Net debt/EBITDA": leave to the LLM
        return None
    if "margin" in low:
        if "ebitda" in low:
            return "ebitda_margin", True
        if "operat" in low or "ebit" in low:
            return "operating_margin", True
        if "net" in low:
            return "net_margin", True
        return "gross_margin", True
    if "yield" in low:
        return ("fcf_yield" if "fcf" in low or "free cash" in low else "dividend_yield"), True
    if re.search(r"\beps\b", low):
        return ("adj_eps" if "adj" in low else "eps"), False
    if any(w in low for w in ("revenue", "sales", "turnover")):
        return "revenue", False
    if "ebitda" in low:
        return "ebitda", False
    if "ebit" in low:
        return "ebit", False
    if "free cash flow" in low or "fcf" in low:
        return "free_cash_flow", False
    if "net income" in low or "net profit" in low:
        return "net_income", False
    return None


def _canonicalize_metric_names(raw_names: list[str]) -> dict[str, tuple[str, bool]]:
    """Map raw Excel row labels to the canonical taxonomy: heuristics first, then ONE batched LLM call."""
    out: dict[str, tuple[str, bool]] = {}
    unknown = []
    for rn in raw_names:
        hit = _heuristic_metric(rn)
        if hit:
            out[rn] = hit
        else:
            unknown.append(rn)
    for i in range(0, len(unknown), 150):
        batch = unknown[i : i + 150]
        system = (
            "Map raw financial metric names from Excel models to this canonical taxonomy: "
            + ", ".join(_CANONICAL_METRICS)
            + ". Also decide if each metric is a ratio (margin, yield, %, ROIC, ROE, ...). Respond ONLY with a JSON "
            'object {"<raw name>": {"canonical": "...", "is_ratio": true|false}, ...}, no code fences.'
        )
        try:
            content = chat(
                [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(batch)}],
                model=INGEST_MODEL,
                temperature=0.0,
            )
            data = json.loads(strip_code_fences(content))
        except Exception:
            data = {}
        for rn in batch:
            d = data.get(rn) if isinstance(data, dict) else None
            canonical = str((d or {}).get("canonical", "other")).strip().lower()
            out[rn] = (canonical if canonical in _CANONICAL_METRICS else "other", bool((d or {}).get("is_ratio")))
    return out


def _extract_metrics_from_excel(path: Path) -> list[dict]:
    """Pragmatic parser: first column = metric label, other columns = periods (2023, FY24E, ...)."""
    try:
        xls = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        return [{"error": f"read_excel failed: {e}"}]

    sheets = []
    raw_names: set[str] = set()
    for df in xls.values():
        if df is None or df.empty:
            continue
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        sheets.append(df)
        raw_names.update(v.strip() for v in df[df.columns[0]] if isinstance(v, str) and v.strip())

    canonical_map = _canonicalize_metric_names(sorted(raw_names))
    metrics: list[dict] = []
    for df in sheets:
        first_col = df.columns[0]
        for _, row in df.iterrows():
            raw_name = row.get(first_col)
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            raw_name = raw_name.strip()
            canonical_name, is_ratio = canonical_map.get(raw_name, ("other", False))
            for col in df.columns[1:]:
                val = row.get(col)
                if isinstance(val, bool) or not isinstance(val, (int, float, np.number)) or pd.isna(val):
                    continue
                metrics.append(
                    {
                        "raw_name": raw_name,
                        "canonical_name": canonical_name,
                        "period": str(col).strip(),
                        "value": float(val),
                        "currency": None,
                        "unit": None,
                        "is_ratio": is_ratio,
                    }  # fmt: skip
                )
    return metrics


def add_models(paths: list[Path]) -> dict[str, Any]:
    """Ingest Excel analyst models (.xlsx/.xlsm) into the models + metrics tables."""
    added = skipped = 0
    details: list[dict] = []
    with _INGEST_LOCK:
        conn = get_conn()
        for p in paths:
            if p.suffix.lower() not in (".xlsx", ".xlsm"):
                details.append({"file": p.name, "status": "ignored", "reason": "not an Excel model"})
                continue
            try:
                raw_bytes = p.read_bytes()
                h = hash_bytes(raw_bytes)
                if conn.execute("SELECT id FROM models WHERE hash=?", (h,)).fetchone():
                    skipped += 1
                    details.append({"file": p.name, "status": "skipped_duplicate", "hash": h})
                    continue
                stored_path = MODELS_DIR / f"{h[:10]}_{p.name}"
                stored_path.write_bytes(raw_bytes)

                companies = _extract_company_names_from_excel(stored_path)
                if not companies:
                    companies = classify_companies(p.stem, model=INGEST_MODEL)

                parsed = _extract_metrics_from_excel(stored_path)
                warnings = [m["error"] for m in parsed if "error" in m]
                rows = [m for m in parsed if "error" not in m]

                cur = conn.execute(
                    "INSERT INTO models(filename, hash, companies, stored_path) VALUES(?,?,?,?)",
                    (p.name, h, json.dumps(companies), str(stored_path)),
                )
                model_id = cur.lastrowid
                conn.executemany(
                    "INSERT INTO metrics(model_id, company, raw_name, canonical_name, period, value, currency, unit, "
                    "is_ratio) VALUES(?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            model_id,
                            c,
                            m["raw_name"],
                            m["canonical_name"],
                            m["period"],
                            m["value"],
                            m["currency"],
                            m["unit"],
                            int(m["is_ratio"]),
                        )
                        for m in rows
                        for c in (companies or [""])
                    ],  # fmt: skip
                )
                conn.commit()
                added += 1
                details.append(
                    {
                        "file": p.name,
                        "status": "added",
                        "hash": h,
                        "companies": companies,
                        "n_metrics": len(rows) * max(1, len(companies)),
                        "warnings": warnings,
                    }  # fmt: skip
                )
            except sqlite3.IntegrityError as e:
                conn.rollback()
                skipped += 1
                details.append({"file": p.name, "status": "skipped_duplicate", "note": str(e)})
            except Exception as e:
                conn.rollback()
                details.append({"file": p.name, "status": "error", "error": f"{type(e).__name__}: {e}"})
        conn.close()
    return {"added": added, "skipped": skipped, "details": details}
