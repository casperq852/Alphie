# db.py
import os
import sqlite3
import shutil
from pathlib import Path

# ---- Original behavior baseline ----
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (BASE_DIR / "data")  # original default
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- New: allow override via env for Unraid bind mounts ----
ENV_DATA_DIR = os.getenv("DATA_DIR")
DATA_DIR = Path(ENV_DATA_DIR).resolve() if ENV_DATA_DIR else DEFAULT_DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Optional one-time migration:
# If user switched to DATA_DIR and the old DB exists but the new one doesn't, copy it over.
old_db = DEFAULT_DATA_DIR / "app.db"
DB_PATH = DATA_DIR / "app.db"
if ENV_DATA_DIR and old_db.exists() and not DB_PATH.exists():
    try:
        # copy DB file
        shutil.copy2(old_db, DB_PATH)
        # best effort: copy known subfolders (vectors/docs/tmp) if present
        for sub in ("vector_store", "docs", "tmp_upload", "models"):
            src = DEFAULT_DATA_DIR / sub
            dst = DATA_DIR / sub
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
        print(f"[db.py] Migrated data from {DEFAULT_DATA_DIR} to {DATA_DIR}")
    except Exception as e:
        print(f"[db.py] Migration warning: {e}")


def _ensure_columns(conn: sqlite3.Connection):
    # documents table migrations
    cols_docs = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
    if "stored_path" not in cols_docs:
        try:
            conn.execute("ALTER TABLE documents ADD COLUMN stored_path TEXT")
            conn.commit()
        except Exception:
            pass
    if "doc_date" not in cols_docs:
        try:
            conn.execute("ALTER TABLE documents ADD COLUMN doc_date TEXT")  # ISO date (YYYY-MM-DD) or NULL
            conn.commit()
        except Exception:
            pass
    if "meta_json" not in cols_docs:
        try:
            conn.execute("ALTER TABLE documents ADD COLUMN meta_json TEXT")
            conn.commit()
        except Exception:
            pass

    # chunks table migrations
    cols_chunks = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
    if "chunk_date" not in cols_chunks:
        try:
            conn.execute("ALTER TABLE chunks ADD COLUMN chunk_date TEXT")  # ISO date or NULL
            conn.commit()
        except Exception:
            pass

    # helpful indices
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_date ON documents(doc_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_date ON chunks(chunk_date)")
        conn.commit()
    except Exception:
        pass

    # ---- New tables / indices for models & metrics ----
    cols_models = {r[1] for r in conn.execute("PRAGMA table_info(models)")}
    if not cols_models:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            hash TEXT UNIQUE,
            companies TEXT,
            stored_path TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()

    cols_metrics = {r[1] for r in conn.execute("PRAGMA table_info(metrics)")}
    if not cols_metrics:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            company TEXT,
            raw_name TEXT,
            canonical_name TEXT,
            period TEXT,
            value REAL,
            currency TEXT,
            unit TEXT,
            is_ratio INTEGER,
            FOREIGN KEY(model_id) REFERENCES models(id)
        )
        """)
        conn.commit()

    # indices for metrics
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_company ON metrics(company)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_canonical ON metrics(canonical_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_model ON metrics(model_id)")
        conn.commit()
    except Exception:
        pass


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))

    # Core RAG tables (existing)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        hash TEXT UNIQUE,
        companies TEXT,
        n_pages INTEGER,
        stored_path TEXT,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        doc_date TEXT,
        meta_json TEXT
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        company TEXT,
        chunk_index INTEGER,
        text TEXT,
        embedding BLOB,
        chunk_date TEXT,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_chunks_company ON chunks(company)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)""")

    # New tables / migrations
    _ensure_columns(conn)
    return conn


def list_documents():
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT d.id, d.filename, d.hash, d.companies, d.n_pages,
               (SELECT COUNT(*) FROM chunks c WHERE c.document_id = d.id) AS n_chunks,
               d.added_at, d.stored_path, d.doc_date
        FROM documents d
        ORDER BY COALESCE(d.doc_date, d.added_at) DESC, d.added_at DESC
        """
    ).fetchall()
    conn.close()
    cols = ["id", "filename", "hash", "companies", "n_pages", "n_chunks", "added_at", "stored_path", "doc_date"]
    return [dict(zip(cols, r)) for r in rows]

# Expose for other modules
__all__ = ["DATA_DIR", "DB_PATH", "get_conn", "list_documents"]
