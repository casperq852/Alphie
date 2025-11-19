import os, io, zipfile, json, base64, time, uuid, html
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag import add_documents, classify_companies, add_models
from db import DATA_DIR, list_documents
from agent_policy import (
    agent_investment_case,
    summarize_events_for_ticker,
)
from custom_query import run_custom_query

# -------------------------------------------------
# Streamlit page config & styling
# -------------------------------------------------
st.set_page_config(page_title="Alphie – Jouw AI-Analist", page_icon="📈", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDataFrame"] table { font-size: 0.85rem; }
small, .caption { opacity:.75 }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.copyline { margin:.25rem 0 .75rem 0; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("""
<div style="display:flex; align-items:center; gap:.6rem; margin-bottom:.25rem;">
  <span style="font-size:2.2rem;">📈</span>
  <span style="font-size:2rem; font-weight:600;">Alphie – Jouw AI-Analist</span>
</div>
<div style="margin-top:-.25rem; opacity:.70;">
  Ondersteund door je eigen researchdocumenten & optionele web data.
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PDF / Web source rendering helpers
# -------------------------------------------------
def _short(p: str, maxlen: int = 48) -> str:
    base = os.path.basename(p)
    return base if len(base) <= maxlen else base[:maxlen-3] + "..."

def _pdf_data_link(path: str, download_name: str) -> str:
    b = Path(path).read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return f'data:application/pdf;base64,{b64}', download_name

def _render_pdf_sources(legend: list[tuple[str, str, str]]) -> str:
    by_path = {}
    for tag, path, name in legend:
        key = path or f"__no_path__::{name}"
        if key not in by_path:
            by_path[key] = {"tags": [], "path": path, "name": name}
        by_path[key]["tags"].append(tag)

    links = []
    for info in by_path.values():
        tags_txt = ",".join(info["tags"])
        name = info["name"]; path = info["path"]
        if path and Path(path).exists():
            href, dl_name = _pdf_data_link(path, name)
            links.append(f'<a href="{href}" download="{dl_name}">[{tags_txt}] {_short(name)}</a>')
        else:
            links.append(
                f'<span title="File not available">[{tags_txt}] {_short(name)}</span>'
            )
    return " · ".join(links)

def _render_web_sources(legend: list[tuple[str, str, str]]) -> str:
    links = []
    for tag, title, url in legend:
        safe_title = (title or url or "link")[:80]
        if url:
            links.append(
                f'<a href="{url}" target="_blank" rel="noopener">[{tag}] {safe_title}</a>'
            )
        else:
            links.append(f'<span>[{tag}] {safe_title}</span>')
    return " · ".join(links)

# -------------------------------------------------
# Sidebar: STATS ONLY
# -------------------------------------------------
st.sidebar.header("Statistieken")

def _db_stats():
    docs = list_documents()
    n_docs = len(docs)
    n_chunks = sum(d.get("n_chunks", 0) for d in docs)
    companies = set()
    lastUpload = None

    for r in docs:
        try:
            comps = json.loads(r.get("companies") or "[]")
            companies.update(comps)
        except Exception:
            pass
        if r.get("added_at"):
            lastUpload = max(lastUpload, r["added_at"]) if lastUpload else r["added_at"]

    return n_docs, n_chunks, len(companies), lastUpload

n_docs, n_chunks, n_companies, lastUpload = _db_stats()

st.sidebar.write(f"**Aantal documenten:** {n_docs}")
st.sidebar.write(f"**Aantal chunks (text):** {n_chunks}")
st.sidebar.write(f"**Aantal bedrijven:** {n_companies}")
st.sidebar.write(f"**Laatste upload:** {lastUpload or 'n.v.t.'}")

st.sidebar.markdown("---")
st.sidebar.write("OPENAI_MODEL:", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab_case, tab_custom, tab_ingest, tab_repo, tab_events = st.tabs(
    ["Case Builder", "Custom Query", "Uploaden", "Repository", "Events / Watchlist"]
)


# -------------------------------------------------
# CASE BUILDER TAB
# -------------------------------------------------
with tab_case:
    st.subheader("Investment Case Builder")

    query = st.text_input("Welk bedrijf wil je analyseren?", "")

    guidance = st.text_area(
        "Aanvullende guidance (optioneel)",
        placeholder="Bijv. focus op margeherstel, recente M&A, risico op regulatie, competitive dynamics, etc.",
        height=80,
    )

    use_web = st.checkbox(
        "Web toestaan (Tavily)",
        value=bool(os.getenv("TAVILY_API_KEY")),
        key="case_builder_web",
    )

    if st.button("Genereer Investment Case", type="primary"):
        if not query.strip():
            st.warning("Voer een vraag of bedrijfsnaam in.")
        else:
            with st.spinner("Analyseren…"):
                out = agent_investment_case(
                    user_query=query,
                    company_filter=None,         # no more manual filter
                    web_allowed=use_web,
                    guidance=guidance,           # NEW FIELD
                )

            st.markdown(out["content"])

            if out.get("legend"):
                html_links = _render_pdf_sources(out["legend"])
                st.markdown(
                    f"<p style='opacity:0.75;font-size:0.9em'>Bronnen: {html_links}</p>",
                    unsafe_allow_html=True,
                )
            if out.get("web_legend"):
                htmlw = _render_web_sources(out["web_legend"])
                st.markdown(
                    f"<p style='opacity:0.75;font-size:0.9em'>Web: {htmlw}</p>",
                    unsafe_allow_html=True,
                )


# -------------------------------------------------
# CUSTOM QUERY TAB (unchanged except removal of model toggles)
# -------------------------------------------------
with tab_custom:
    st.subheader("Custom Query Mode (Advanced)")
    st.caption("Vrije analyse met volledige controle over parameters.")

    model_choice = st.radio(
        "Model",
        ["gpt-4o-mini", "gpt-5-mini"],
        index=0
    )

    # Web
    use_web_cq = st.checkbox("Web toestaan (Tavily)", value=bool(os.getenv("TAVILY_API_KEY")), key="custom_query_web")

    # Chunk retrieval parameters
    k_chunks = st.number_input("Aantal PDF-chunks", min_value=4, max_value=200, value=18, step=2)

    # Date filter
    min_date = st.text_input("Minimale datum (ISO; optioneel)", value="")
    min_date = min_date or None

    # Token budget
    max_tokens = st.slider("Max tokens", min_value=400, max_value=8000, value=2000, step=200)

    detail_choice = "exhaustive"

    focus_notes = st.text_area("Focus / parameters", height=140)

    cq_query = st.text_area("Jouw vraag", height=180)

    if st.button("Analyseer", type="primary"):
        with st.spinner("Analyseren…"):
            out = run_custom_query(
                focus_notes=focus_notes,
                user_query=cq_query,
                use_web=use_web_cq,
                model=model_choice,
                k_chunks=k_chunks,
                min_date=min_date,
                detail=detail_choice,
                max_tokens=max_tokens,
            )

        st.markdown(out["content"])

        if out.get("legend"):
            html_links = _render_pdf_sources(out["legend"])
            st.markdown(f"<p style='opacity:0.75;font-size:0.9em'>Bronnen: {html_links}</p>",
                        unsafe_allow_html=True)
        if out.get("web_legend"):
            htmlw = _render_web_sources(out["web_legend"])
            st.markdown(f"<p style='opacity:0.75;font-size:0.9em'>Web: {htmlw}</p>",
                        unsafe_allow_html=True)

        with st.expander("Debug"):
            st.write({
                "Detected companies": out.get("companies"),
                "Raw hits": out.get("raw_hits"),
                "Filtered hits": out.get("filtered_hits"),
                "Web debug": out.get("web_debug"),
            })


# -------------------------------------------------
# UPLOAD TAB (EXTENDED: PDFs + Excel models)
# -------------------------------------------------
with tab_ingest:
    st.subheader("Documenten Uploaden")
    st.caption("Upload researchrapporten als PDF of ZIP, en analistenmodellen als Excel (.xlsx).")

    TMP_UPLOAD_DIR = DATA_DIR / "tmp_upload"
    TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    up_zip = st.file_uploader("Upload ZIP", type=["zip"])
    up_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    up_xlsx = st.file_uploader("Upload Excel modellen", type=["xlsx"], accept_multiple_files=True)

    to_process_pdfs: list[Path] = []
    to_process_models: list[Path] = []

    # ZIP handling: accept both PDFs and Excel models inside
    if up_zip:
        with zipfile.ZipFile(io.BytesIO(up_zip.read())) as z:
            names = [
                n for n in z.namelist()
                if n.lower().endswith(".pdf") or n.lower().endswith(".xlsx")
            ]
            total = len(names)
            if total == 0:
                st.warning("Geen PDFs of Excel-bestanden in ZIP.")
            else:
                tmpdir = TMP_UPLOAD_DIR
                prog = st.progress(0.0)
                for i, name in enumerate(names, start=1):
                    dest = tmpdir / Path(name).name
                    with z.open(name) as f, open(dest, "wb") as out:
                        out.write(f.read())
                    if dest.suffix.lower() == ".pdf":
                        to_process_pdfs.append(dest)
                    elif dest.suffix.lower() == ".xlsx":
                        to_process_models.append(dest)
                    prog.progress(i / total)
                st.success(f"{total} bestand(en) uitgepakt.")

    # Direct PDFs
    if up_pdfs:
        tmpdir = TMP_UPLOAD_DIR
        for f in up_pdfs:
            dest = tmpdir / f.name
            with open(dest, "wb") as out:
                out.write(f.read())
            to_process_pdfs.append(dest)

    # Direct Excel models
    if up_xlsx:
        tmpdir = TMP_UPLOAD_DIR
        for f in up_xlsx:
            dest = tmpdir / f.name
            with open(dest, "wb") as out:
                out.write(f.read())
            to_process_models.append(dest)

    # Process PDFs
    if to_process_pdfs:
        st.info("PDF-documenten worden verwerkt…")
        prog = st.progress(0.0)
        added = skipped = 0
        details = []
        total = len(to_process_pdfs)

        for i, p in enumerate(to_process_pdfs, start=1):
            try:
                res = add_documents([Path(p)])
                added += res.get("added", 0)
                skipped += res.get("skipped", 0)
                details.extend(res.get("details", []))
            except Exception as e:
                details.append({"file": p.name, "status": "error", "error": str(e)})
            prog.progress(i / total)

        st.success(f"PDFs toegevoegd: {added} • Overgeslagen: {skipped}")
        with st.expander("Details (PDFs)"):
            st.json({"added": added, "skipped": skipped, "details": details})

    # Process Excel models
    if to_process_models:
        st.info("Excel-modellen worden verwerkt…")
        prog = st.progress(0.0)
        added_m = skipped_m = 0
        details_m = []
        total_m = len(to_process_models)

        for i, p in enumerate(to_process_models, start=1):
            try:
                res = add_models([Path(p)])
                added_m += res.get("added", 0)
                skipped_m += res.get("skipped", 0)
                details_m.extend(res.get("details", []))
            except Exception as e:
                details_m.append({"file": p.name, "status": "error", "error": str(e)})
            prog.progress(i / total_m)

        st.success(f"Modellen toegevoegd: {added_m} • Overgeslagen: {skipped_m}")
        with st.expander("Details (Excel-modellen)"):
            st.json({"added": added_m, "skipped": skipped_m, "details": details_m})


# -------------------------------------------------
# REPOSITORY TAB
# -------------------------------------------------
with tab_repo:
    st.subheader("Documentenbibliotheek")

    @st.cache_data(ttl=60)
    def _cached_docs(_nonce: int) -> list[dict]:
        return list_documents()

    docs = _cached_docs(int(time.time() // 60))
    if not docs:
        st.info("Nog geen documenten.")
    else:
        q = st.text_input("Filter", "")
        view = docs
        if q.strip():
            ql = q.lower()
            def ok(row):
                if ql in str(row["filename"]).lower(): return True
                if ql in str(row["companies"]).lower(): return True
                if row.get("doc_date") and ql in str(row["doc_date"]).lower(): return True
                return False
            view = [r for r in docs if ok(r)]

        for r in view:
            try:
                lst = json.loads(r["companies"]) if r["companies"] else []
            except Exception:
                lst = []
            r["companies"] = ", ".join(lst)

        st.dataframe(view, use_container_width=True, hide_index=True)


# -------------------------------------------------
# EVENTS / WATCHLIST TAB
# -------------------------------------------------
with tab_events:
    st.subheader("Watchlist & Event Updates")

    WATCHLIST_PATH = DATA_DIR / "watchlist.json"

    def _load_watchlists():
        if WATCHLIST_PATH.exists():
            raw = json.loads(WATCHLIST_PATH.read_text())
            if isinstance(raw, dict) and "lists" in raw:
                lists = raw.get("lists", {})
                if not lists:
                    lists = {"Default": ["ASML NA", "AAPL US"]}
                active = raw.get("active") or next(iter(lists))
                return lists, active
        return {"Default": ["ASML NA", "AAPL US"]}, "Default"

    def _save_watchlists(watchlists, active):
        WATCHLIST_PATH.write_text(json.dumps({"lists": watchlists, "active": active}, indent=2))

    watchlists, active_watchlist_name = _load_watchlists()

    col_w1, col_w2 = st.columns([2, 1])
    with col_w1:
        watchlist_names = sorted(watchlists.keys())
        if active_watchlist_name not in watchlist_names:
            active_watchlist_name = watchlist_names[0]
        selected_name = st.selectbox(
            "Kies watchlist",
            watchlist_names,
            index=watchlist_names.index(active_watchlist_name),
            key="watchlist_select",
        )

    with col_w2:
        new_name = st.text_input("Nieuwe lijst", "", key="new_watchlist_name")
        if st.button("Create / reset lijst", use_container_width=True):
            name = new_name.strip()
            if name:
                watchlists[name] = []
                active_watchlist_name = name
                _save_watchlists(watchlists, active_watchlist_name)
                st.success(f"Lijst '{name}' opgeslagen.")

    if selected_name:
        tickers = watchlists.get(selected_name, [])
        st.markdown(f"**Tickers in '{selected_name}':** {', '.join(tickers) or '(geen)'}")

        ticker_to_add = st.text_input("Voeg ticker toe", "")
        if st.button("Toevoegen"):
            if ticker_to_add.strip():
                watchlists[selected_name].append(ticker_to_add.strip())
                _save_watchlists(watchlists, selected_name)
                st.success("Toegevoegd.")

        if tickers:
            st.markdown("---")
            st.subheader("Event updates genereren")
            days = st.slider("Periode (dagen)", min_value=5, max_value=90, value=14)

            if st.button("Genereer updates"):
                for tkr in tickers:
                    with st.spinner(f"{tkr}…"):
                        out = summarize_events_for_ticker(tkr, days=days)
                    st.markdown(out["content"])
                    if out.get("doc_legend"):
                        html_links = _render_pdf_sources(out["doc_legend"])
                        st.markdown(
                            f"<p style='opacity:0.75;font-size:0.9em'>Bronnen: {html_links}</p>",
                            unsafe_allow_html=True,
                        )
                    if out.get("web_legend"):
                        htmlw = _render_web_sources(out["web_legend"])
                        st.markdown(
                            f"<p style='opacity:0.75;font-size:0.9em'>Web: {htmlw}</p>",
                            unsafe_allow_html=True,
                        )
