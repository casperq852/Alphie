from datetime import date, timedelta
from pathlib import Path

import pytest

import rag


@pytest.fixture
def repo(offline, monkeypatch, tmp_path):
    rag = offline
    docs = {}

    def fake_pdf_to_text(path: Path):
        return docs[path.name.split("_", 1)[1]], {}

    monkeypatch.setattr(rag, "pdf_to_text", fake_pdf_to_text)
    company_of = {}
    monkeypatch.setattr(rag, "classify_companies", lambda text, model=None: company_of[text])

    def add(name, company, text, doc_date):
        docs[name] = [f"{doc_date}\n{text}"]
        company_of[f"{doc_date}\n{text}"] = [company]
        p = tmp_path / name
        p.write_bytes(name.encode() + text.encode())
        return rag.add_documents([p])

    return rag, add


def test_company_filter_has_full_recall(repo):
    rag, add = repo
    today = date.today().isoformat()
    for i in range(30):
        add(f"asml_{i}.pdf", "ASML", f"lithography euv margin outlook note {i} " * 20, today)
    add("giv.pdf", "Givaudan", "fragrance flavours volume growth pricing", today)

    # query semantically closer to ASML docs, but filtered on Givaudan -> must still find the Givaudan chunk
    hits = rag.search_chunks("euv lithography margin outlook pricing", k=5, company_filter=["Givaudan"])
    assert hits and all(h["source"] == "giv.pdf" for h in hits)

    unfiltered = rag.search_chunks("euv lithography margin outlook", k=5)
    assert unfiltered and unfiltered[0]["source"].startswith("asml_")


def test_duplicate_skipped_and_date_window(repo):
    rag, add = repo
    old = (date.today() - timedelta(days=200)).isoformat()
    assert add("old.pdf", "Heineken", "beer volumes premiumisation", old)["added"] == 1
    assert add("old.pdf", "Heineken", "beer volumes premiumisation", old)["skipped"] == 1
    assert rag.search_chunks("beer volumes", company_filter=["Heineken"], date_from=rag.days_ago_iso(14)) == []
    assert rag.search_chunks("beer volumes", company_filter=["Heineken"], date_from=rag.days_ago_iso(365))


def test_delete_and_rebuild(repo):
    rag, add = repo
    add("del.pdf", "Unilever", "soap ice cream spin off", date.today().isoformat())
    doc = next(d for d in rag.get_conn().execute("SELECT id, filename FROM documents") if d[1] == "del.pdf")
    assert rag.delete_document(doc[0])
    assert all(h["source"] != "del.pdf" for h in rag.search_chunks("soap ice cream spin off", k=50))
    _, meta = rag._load_index_and_meta()
    assert len(meta["ids"]) == rag.get_conn().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def test_metric_heuristics():
    assert rag._heuristic_metric("EBITDA margin") == ("ebitda_margin", True)
    assert rag._heuristic_metric("Adj. EPS") == ("adj_eps", False)
    assert rag._heuristic_metric("Net debt/EBITDA") is None
    assert rag._heuristic_metric("Deposits") is None
