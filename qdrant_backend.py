"""Optional Qdrant vector backend. Returns raw similarity hits; score shaping happens in rag.py."""

import os
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from llm import EMB_DIM

QDRANT_URL = os.getenv("QDRANT_URL", "http://10.10.30.18:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "research_chunks")
_UPSERT_BATCH = 64


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)
    return QdrantClient(url=QDRANT_URL, timeout=30.0)


@lru_cache(maxsize=1)
def ensure_collection() -> None:
    c = get_client()
    try:
        exists = c.collection_exists(QDRANT_COLLECTION)
    except Exception:  # older clients
        try:
            c.get_collection(QDRANT_COLLECTION)
            exists = True
        except Exception:
            exists = False
    if not exists:
        c.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qm.VectorParams(size=EMB_DIM, distance=qm.Distance.COSINE),
            optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=20000),
        )
    try:  # makes document_id-filtered search fast; idempotent
        c.create_payload_index(QDRANT_COLLECTION, field_name="document_id", field_schema=qm.PayloadSchemaType.INTEGER)
    except Exception:
        pass


def upsert_chunks(
    doc_id: int,
    vectors: list[list[float]],
    chunks: list[str],
    companies: list[str],
    stored_path: str,
    doc_date: str | None,
    chunk_dates: list[str | None],
) -> None:
    ensure_collection()
    points = []
    for i, (v, txt, cdate) in enumerate(zip(vectors, chunks, chunk_dates, strict=True)):
        payload = {
            "document_id": doc_id,
            "chunk_index": i,
            "text": txt,
            "companies": companies,
            "stored_path": stored_path,
            "doc_date": doc_date,
            "chunk_date": cdate,
        }
        # Deterministic id so re-ingesting the same chunk overwrites instead of duplicating
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{doc_id}/chunk:{i}"))
        points.append(qm.PointStruct(id=pid, vector=v, payload=payload))
    for i in range(0, len(points), _UPSERT_BATCH):
        get_client().upsert(collection_name=QDRANT_COLLECTION, points=points[i : i + _UPSERT_BATCH])


def delete_document(doc_id: int) -> None:
    ensure_collection()
    get_client().delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=doc_id))])
        ),
    )


def search(query_vec: list[float], limit: int, doc_ids: list[int] | None = None) -> list[dict]:
    """Vector search, optionally restricted to a set of document ids (company / date pre-filter)."""
    ensure_collection()
    flt = None
    if doc_ids is not None:
        flt = qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchAny(any=list(doc_ids)))])
    hits = get_client().search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vec,
        limit=limit,
        query_filter=flt,
        with_payload=True,
    )
    out = []
    for h in hits:
        p = h.payload or {}
        out.append(
            {
                "score": float(h.score or 0.0),
                "text": p.get("text", ""),
                "companies": p.get("companies", []),
                "document_id": p.get("document_id"),
                "chunk_index": p.get("chunk_index"),
                "stored_path": p.get("stored_path"),
                "chunk_date": p.get("chunk_date"),
                "doc_date": p.get("doc_date"),
            }
        )
    return out
