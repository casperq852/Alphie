# qdrant_backend.py
import os, json, math
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import uuid

QDRANT_URL = os.getenv("QDRANT_URL", "http://10.10.30.18:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "research_chunks")
EMB_DIM = 3072  # matches text-embedding-3-large

def get_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)
    return QdrantClient(url=QDRANT_URL, timeout=30.0)

def ensure_collection():
    c = get_client()
    try:
        # Preferred (newer clients)
        exists = c.collection_exists(QDRANT_COLLECTION)
    except Exception:
        # Fallback (older clients don't have collection_exists)
        try:
            c.get_collection(QDRANT_COLLECTION)  # will raise if missing
            exists = True
        except Exception:
            exists = False

    if not exists:
        c.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=EMB_DIM, distance=qm.Distance.COSINE),
            optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=20000),
        )

def upsert_chunks(
    doc_id: int,
    vectors: List[List[float]],
    chunks: List[str],
    companies_json: str,
    stored_path: str,
    doc_date: Optional[str],
):
    ensure_collection()
    companies = []
    try:
        companies = json.loads(companies_json or "[]")
    except Exception:
        pass
    payloads = []
    points = []
    for i, (v, txt) in enumerate(zip(vectors, chunks)):
        payload = {
            "document_id": doc_id,
            "chunk_index": i,
            "text": txt,
            "companies": companies,
            "stored_path": stored_path,
            "doc_date": doc_date,
            "chunk_date": _first_date_in_text(txt),
        }
        # Use a deterministic UUID so re-ingests upsert the same chunk
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{doc_id}/chunk:{i}"))
        points.append(qm.PointStruct(id=pid, vector=v, payload=payload))

    get_client().upsert(collection_name=QDRANT_COLLECTION, points=points)

def _first_date_in_text(text: str) -> Optional[str]:
    import re, dateparser
    # loose scan; same spirit as your rag.py
    MONTHS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)"
    pats = [
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        rf"\b\d{{1,2}}\s+{MONTHS}\s+\d{{4}}\b",
        rf"\b{MONTHS}\s+\d{{1,2}},\s*\d{{4}}\b",
    ]
    for pat in pats:
        m = re.search(pat, text or "", flags=re.IGNORECASE)
        if m:
            d = dateparser.parse(m.group(0))
            if d: return d.date().isoformat()
    return None

def _norm(s: str) -> str:
    import re
    s = (s or "").lower()
    s = re.sub(r"\b(group|plc|inc|nv|sa|ltd|corp|co|ag|se|nv)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def search(
    query_vec: List[float],
    k: int = 12,
    company_filter: Optional[List[str]] = None,
    date_focus: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None),
):
    ensure_collection()
    cf_norm = [_norm(c) for c in (company_filter or [])]

    # Build filter
    must: List[qm.FieldCondition] = []
    # company filter: any overlap
    if cf_norm:
        must.append(
            qm.FieldCondition(
                key="companies",
                match=qm.MatchAny(any=company_filter)  # raw strings; we'll soft-screen post-query too
            )
        )

    # vector search
    hits = get_client().search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vec,
        limit=k * 4,
        score_threshold=None,
        query_filter=qm.Filter(must=must) if must else None,
        with_payload=True,
    )

    results = []
    exact_date, date_from, date_to = date_focus
    for h in hits:
        p = h.payload or {}
        txt = p.get("text", "")
        comps = p.get("companies", [])
        chunk_date = p.get("chunk_date")
        doc_date = p.get("doc_date")
        stored_path = p.get("stored_path")
        document_id = p.get("document_id")
        chunk_index = p.get("chunk_index")
        score = float(h.score or 0.0)

        # Soft company screen to tolerate variants
        if cf_norm:
            comp_norms = [_norm(c) for c in comps]
            ok = any(any(f in c for c in comp_norms) for f in cf_norm)
            if not ok:
                continue

        # Light recency / date focus shaping (mirrors your rag.py)
        from datetime import datetime
        base_date = None
        for d in (chunk_date, doc_date):
            if d:
                try:
                    base_date = datetime.fromisoformat(str(d))
                    break
                except Exception:
                    pass
        if base_date:
            age_days = max(0, (datetime.now() - base_date).days)
            score *= math.exp(-(age_days / 120.0))

        if exact_date or (date_from and date_to):
            try:
                cd = datetime.fromisoformat(chunk_date) if chunk_date else None
            except Exception:
                cd = None
            if exact_date:
                if cd:
                    from datetime import datetime as dt
                    delta = abs((cd.date() - dt.fromisoformat(exact_date).date()).days)
                    score *= math.exp(-(delta ** 2) / (2 * (2.0 ** 2))) * 1.5
                else:
                    score *= 0.85
            else:
                # range bonus
                from datetime import datetime as dt
                try:
                    f = dt.fromisoformat(date_from).date()
                    t = dt.fromisoformat(date_to).date()
                    if cd and f <= cd.date() <= t:
                        score *= 1.4
                    else:
                        score *= 0.9
                except Exception:
                    pass

        results.append({
            "score": score, "text": txt, "companies": comps,
            "document_id": document_id, "chunk_index": chunk_index,
            "source": "", "stored_path": stored_path,
            "chunk_date": chunk_date, "doc_date": doc_date
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]
