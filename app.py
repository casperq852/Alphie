import html
import io
import json
import os
import re
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # must run before the modules below read their env config

from agent_policy import agent_investment_case, summarize_events_for_ticker  # noqa: E402
from custom_query import run_custom_query  # noqa: E402
from db import DATA_DIR, list_documents, list_models  # noqa: E402
from llm import AVAILABLE_MODELS, INGEST_MODEL, OPENAI_MODEL  # noqa: E402
from rag import USE_QDRANT, add_documents, add_models, delete_document, rebuild_faiss_index  # noqa: E402
from websearch import TAVILY_API_KEY  # noqa: E402

st.set_page_config(page_title="Alphie – Jouw AI-Analist", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDataFrame"] table { font-size: 0.85rem; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.srcline { opacity: .75; font-size: .9em; }
</style>
<div style="display:flex; align-items:center; gap:.6rem; margin-bottom:.25rem;">
  <span style="font-size:2.2rem;">📈</span>
  <span style="font-size:2rem; font-weight:600;">Alphie – Jouw AI-Analist</span>
</div>
<div style="margin-top:-.25rem; opacity:.70;">Ondersteund door je eigen researchdocumenten & optionele web data.</div>
""",
    unsafe_allow_html=True,
)

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY ontbreekt. Zet hem in `.env` (zie `.env.example`) en herstart de app.")
    st.stop()

TMP_UPLOAD_DIR = DATA_DIR / "tmp_upload"
TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
WATCHLIST_PATH = DATA_DIR / "watchlist.json"
WEB_DEFAULT = bool(TAVILY_API_KEY)


# -------------------------------------------------
# Rendering helpers
# -------------------------------------------------
def _short(name: str, maxlen: int = 60) -> str:
    base = os.path.basename(name or "")
    return base if len(base) <= maxlen else base[: maxlen - 3] + "..."


def _safe_url(url: str) -> str:
    u = (url or "").strip()
    return html.escape(u, quote=True) if u.startswith(("http://", "https://")) else ""


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s or "").strip("_")[:60] or "output"


def render_sources(legend: list, web_legend: list, key: str) -> None:
    """PDF sources as download buttons (grouped per document), web sources as links."""
    if legend:
        by_path: dict[str, dict] = {}
        for tag, path, name in legend:
            info = by_path.setdefault(path or f"__nopath__{name}", {"tags": [], "path": path, "name": name})
            info["tags"].append(tag)
        with st.expander(f"PDF-bronnen ({len(by_path)} documenten)"):
            for i, info in enumerate(by_path.values()):
                label = f"[{','.join(info['tags'])}] {_short(info['name'])}"
                p = Path(info["path"]) if info["path"] else None
                if p and p.exists():
                    st.download_button(
                        label, p.read_bytes(), file_name=info["name"], mime="application/pdf", key=f"{key}_pdf_{i}"
                    )
                else:
                    st.caption(f"{label} (bestand niet beschikbaar)")
    if web_legend:
        links = []
        for tag, title, url in web_legend:
            t = html.escape((title or url or "link")[:80])
            href = _safe_url(url)
            links.append(f'<a href="{href}" target="_blank" rel="noopener">[{html.escape(tag)}] {t}</a>' if href else
                         f"<span>[{html.escape(tag)}] {t}</span>")  # fmt: skip
        st.markdown(f"<p class='srcline'>Web: {' · '.join(links)}</p>", unsafe_allow_html=True)


def render_result(out: dict, key: str, title: str) -> None:
    st.markdown(out["content"])
    render_sources(out.get("legend") or out.get("doc_legend") or [], out.get("web_legend") or [], key)
    st.download_button(
        "Download als Markdown",
        out["content"],
        file_name=f"{date.today().isoformat()}_{_slug(title)}.md",
        mime="text/markdown",
        key=f"{key}_md",
    )


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
@st.cache_data(ttl=30)
def _db_stats(_nonce: int):
    docs = list_documents()
    companies: set[str] = set()
    for r in docs:
        try:
            companies.update(json.loads(r.get("companies") or "[]"))
        except Exception:
            pass
    last = max((r["added_at"] for r in docs if r.get("added_at")), default=None)
    return len(docs), sum(d.get("n_chunks", 0) for d in docs), len(companies), last


st.session_state.setdefault("db_nonce", 0)
n_docs, n_chunks, n_companies, last_upload = _db_stats(st.session_state["db_nonce"])
st.sidebar.header("Statistieken")
st.sidebar.write(f"**Aantal documenten:** {n_docs}")
st.sidebar.write(f"**Aantal chunks (text):** {n_chunks}")
st.sidebar.write(f"**Aantal bedrijven:** {n_companies}")
st.sidebar.write(f"**Laatste upload:** {last_upload or 'n.v.t.'}")
st.sidebar.markdown("---")
st.sidebar.caption(f"Antwoordmodel: `{OPENAI_MODEL}`")
st.sidebar.caption(f"Ingestiemodel: `{INGEST_MODEL}`")
st.sidebar.caption(f"Vector store: `{'Qdrant' if USE_QDRANT else 'FAISS (lokaal)'}`")
st.sidebar.caption(f"Web (Tavily): `{'aan' if TAVILY_API_KEY else 'geen API key'}`")


def _invalidate_stats() -> None:
    st.session_state["db_nonce"] += 1


tab_case, tab_custom, tab_ingest, tab_repo, tab_events = st.tabs(
    ["Case Builder", "Custom Query", "Uploaden", "Repository", "Events / Watchlist"]
)

# -------------------------------------------------
# CASE BUILDER
# -------------------------------------------------
with tab_case:
    st.subheader("Investment Case Builder")
    with st.form("case_form"):
        query = st.text_input("Welk bedrijf wil je analyseren?", "")
        guidance = st.text_area(
            "Aanvullende guidance (optioneel)",
            placeholder="Bijv. focus op margeherstel, recente M&A, risico op regulatie, competitive dynamics, etc.",
            height=80,
        )
        c1, c2 = st.columns([1, 1])
        use_web = c1.checkbox("Web toestaan (Tavily)", value=WEB_DEFAULT, disabled=not TAVILY_API_KEY)
        case_model = c2.selectbox("Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(OPENAI_MODEL)
                                  if OPENAI_MODEL in AVAILABLE_MODELS else 0)  # fmt: skip
        submitted = st.form_submit_button("Genereer Investment Case", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Voer een vraag of bedrijfsnaam in.")
        else:
            bar = st.progress(0.0, text="Starten…")
            try:
                st.session_state["case_out"] = agent_investment_case(
                    user_query=query, company_filter=None, web_allowed=use_web, guidance=guidance,
                    model_override=case_model, progress=lambda msg, frac: bar.progress(min(frac, 1.0), text=msg),
                )  # fmt: skip
            except Exception as e:
                st.error(f"Mislukt: {type(e).__name__}: {e}")
            bar.empty()

    if out := st.session_state.get("case_out"):
        if not out.get("agent_local_hits"):
            st.info(f"Geen interne documenten gevonden voor **{out.get('company')}**; de case leunt op web/kennis.")
        render_result(out, "case", f"case_{out.get('company', '')}")

# -------------------------------------------------
# CUSTOM QUERY
# -------------------------------------------------
with tab_custom:
    st.subheader("Custom Query Mode (Advanced)")
    st.caption("Vrije analyse met volledige controle over parameters.")
    with st.form("cq_form"):
        c1, c2, c3 = st.columns(3)
        model_choice = c1.selectbox("Model", AVAILABLE_MODELS, index=0)
        k_chunks = c2.number_input("Aantal PDF-chunks", min_value=4, max_value=200, value=18, step=2)
        max_tokens = c3.slider("Max tokens", min_value=400, max_value=8000, value=2000, step=200)
        c4, c5, c6 = st.columns(3)
        use_web_cq = c4.checkbox("Web toestaan (Tavily)", value=WEB_DEFAULT, disabled=not TAVILY_API_KEY)
        min_date_in = c5.date_input("Gepubliceerd vanaf (optioneel)", value=None)
        company_mode = c6.selectbox("Bedrijfsfilter", ["Automatisch", "Handmatig", "Geen (hele repository)"])
        company_text = st.text_input("Bedrijven (bij 'Handmatig', komma-gescheiden)", "")
        focus_notes = st.text_area("Focus / parameters", height=120)
        cq_query = st.text_area("Jouw vraag", height=160)
        cq_submit = st.form_submit_button("Analyseer", type="primary")

    if cq_submit:
        if not cq_query.strip():
            st.warning("Voer een vraag in.")
        else:
            override = None
            if company_mode == "Handmatig":
                override = [c.strip() for c in company_text.split(",") if c.strip()]
            elif company_mode.startswith("Geen"):
                override = []
            with st.spinner("Analyseren…"):
                try:
                    st.session_state["cq_out"] = run_custom_query(
                        focus_notes=focus_notes, user_query=cq_query, use_web=use_web_cq, model=model_choice,
                        k_chunks=int(k_chunks), min_date=min_date_in.isoformat() if min_date_in else None,
                        detail="exhaustive", max_tokens=max_tokens, company_override=override,
                    )  # fmt: skip
                except Exception as e:
                    st.error(f"Mislukt: {type(e).__name__}: {e}")

    if out := st.session_state.get("cq_out"):
        render_result(out, "cq", "custom_query")
        with st.expander("Debug"):
            st.write(
                {
                    "Gebruikte bedrijven": out.get("companies"),
                    "PDF-hits": out.get("local_hits"),
                    "Web debug": out.get("web_debug"),
                }  # fmt: skip
            )

# -------------------------------------------------
# UPLOAD
# -------------------------------------------------
with tab_ingest:
    st.subheader("Documenten Uploaden")
    st.caption("Upload researchrapporten als PDF of ZIP, en analistenmodellen als Excel (.xlsx/.xlsm).")

    with st.form("upload_form", clear_on_submit=True):
        up_zip = st.file_uploader("Upload ZIP", type=["zip"])
        up_pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        up_xlsx = st.file_uploader("Upload Excel modellen", type=["xlsx", "xlsm"], accept_multiple_files=True)
        do_ingest = st.form_submit_button("Verwerk uploads", type="primary")

    if do_ingest:
        pdfs: list[Path] = []
        models: list[Path] = []

        def _stash(name: str, data: bytes) -> None:
            dest = TMP_UPLOAD_DIR / Path(name).name  # basename only: no path traversal from ZIP entries
            dest.write_bytes(data)
            (pdfs if dest.suffix.lower() == ".pdf" else models).append(dest)

        if up_zip:
            try:
                with zipfile.ZipFile(io.BytesIO(up_zip.getvalue())) as z:
                    for n in z.namelist():
                        base = Path(n).name
                        if n.endswith("/") or "__MACOSX" in n or base.startswith("._"):
                            continue
                        if base.lower().endswith((".pdf", ".xlsx", ".xlsm")):
                            _stash(base, z.read(n))
            except zipfile.BadZipFile:
                st.error("Ongeldig ZIP-bestand.")
        for f in up_pdfs or []:
            _stash(f.name, f.getvalue())
        for f in up_xlsx or []:
            _stash(f.name, f.getvalue())

        if not pdfs and not models:
            st.warning("Geen PDF- of Excel-bestanden gevonden.")

        def _run(paths: list[Path], fn, label: str) -> None:
            bar = st.progress(0.0, text=f"{label} verwerken…")
            added = skipped = 0
            details = []
            for i, p in enumerate(paths, start=1):
                bar.progress((i - 1) / len(paths), text=f"{label} {i}/{len(paths)}: {p.name}")
                res = fn([p])
                added += res["added"]
                skipped += res["skipped"]
                details.extend(res["details"])
                p.unlink(missing_ok=True)
            bar.empty()
            errors = [d for d in details if d.get("status") == "error"]
            msg = f"{label} toegevoegd: {added} • Overgeslagen (duplicaat): {skipped} • Fouten: {len(errors)}"
            (st.warning if errors else st.success)(msg)
            with st.expander(f"Details ({label})"):
                st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)

        if pdfs:
            _run(pdfs, add_documents, "PDFs")
        if models:
            _run(models, add_models, "Modellen")
        if pdfs or models:
            _invalidate_stats()

# -------------------------------------------------
# REPOSITORY
# -------------------------------------------------
with tab_repo:
    st.subheader("Documentenbibliotheek")
    docs = list_documents()
    if not docs:
        st.info("Nog geen documenten.")
    else:
        q = st.text_input("Filter (bestandsnaam, bedrijf of datum)", "")
        rows = []
        for r in docs:
            try:
                comps = ", ".join(json.loads(r["companies"] or "[]"))
            except Exception:
                comps = r["companies"] or ""
            rows.append({"id": r["id"], "filename": r["filename"], "companies": comps, "doc_date": r["doc_date"],
                         "pages": r["n_pages"], "chunks": r["n_chunks"], "added_at": r["added_at"]})  # fmt: skip
        df = pd.DataFrame(rows)
        if q.strip():
            ql = q.lower()
            mask = (
                df[["filename", "companies", "doc_date"]]
                .astype(str)
                .apply(lambda s: s.str.lower().str.contains(ql, regex=False))
            )
            df = df[mask.any(axis=1)]
        st.caption(f"{len(df)} van {len(rows)} documenten")
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("Beheer"):
            labels = {f"{r['id']} – {r['filename']}": r["id"] for r in rows}
            to_delete = st.multiselect("Documenten verwijderen", list(labels))
            if st.button("Verwijder geselecteerde", disabled=not to_delete):
                n = sum(delete_document(labels[k]) for k in to_delete)
                _invalidate_stats()
                st.success(f"{n} document(en) verwijderd.")
                st.rerun()
            if not USE_QDRANT and st.button("Herbouw FAISS-index uit database"):
                with st.spinner("Index herbouwen…"):
                    n = rebuild_faiss_index()
                st.success(f"Index herbouwd: {n} chunks.")

    models_list = list_models()
    if models_list:
        st.subheader("Excel-modellen")
        st.dataframe(pd.DataFrame(models_list), use_container_width=True, hide_index=True)

# -------------------------------------------------
# EVENTS / WATCHLIST
# -------------------------------------------------
with tab_events:
    st.subheader("Watchlist & Event Updates")

    def _load_watchlists() -> tuple[dict[str, list[str]], str]:
        default = {"Default": ["ASML NA", "AAPL US"]}
        try:
            raw = json.loads(WATCHLIST_PATH.read_text()) if WATCHLIST_PATH.exists() else {}
        except Exception:
            raw = {}
        lists = raw.get("lists") or default
        active = raw.get("active") if raw.get("active") in lists else next(iter(lists))
        return lists, active

    def _save_watchlists(lists: dict[str, list[str]], active: str) -> None:
        WATCHLIST_PATH.write_text(json.dumps({"lists": lists, "active": active}, indent=2))

    watchlists, active_name = _load_watchlists()
    names = sorted(watchlists)

    c1, c2 = st.columns([2, 1])
    selected = c1.selectbox("Kies watchlist", names, index=names.index(active_name))
    if selected != active_name:
        _save_watchlists(watchlists, selected)
    with c2:
        new_name = st.text_input("Nieuwe lijst", "")
        if st.button("Aanmaken", use_container_width=True, disabled=not new_name.strip()):
            name = new_name.strip()
            if name in watchlists:
                st.warning(f"Lijst '{name}' bestaat al.")
            else:
                watchlists[name] = []
                _save_watchlists(watchlists, name)
                st.rerun()

    tickers = watchlists.get(selected, [])
    st.markdown(f"**Tickers in '{selected}':** {', '.join(tickers) or '(geen)'}")

    c3, c4, c5 = st.columns([2, 2, 1])
    add_txt = c3.text_input("Voeg ticker(s) toe (komma-gescheiden)", "")
    if c3.button("Toevoegen", disabled=not add_txt.strip()):
        for t in (x.strip() for x in add_txt.split(",")):
            if t and t not in tickers:
                tickers.append(t)
        watchlists[selected] = tickers
        _save_watchlists(watchlists, selected)
        st.rerun()
    to_remove = c4.multiselect("Verwijder tickers", tickers)
    if c4.button("Verwijderen", disabled=not to_remove):
        watchlists[selected] = [t for t in tickers if t not in to_remove]
        _save_watchlists(watchlists, selected)
        st.rerun()
    if c5.button("Lijst verwijderen", disabled=len(watchlists) <= 1):
        watchlists.pop(selected, None)
        _save_watchlists(watchlists, next(iter(watchlists)))
        st.rerun()

    if tickers:
        st.markdown("---")
        st.subheader("Event updates genereren")
        c6, c7 = st.columns([3, 1])
        days = c6.slider("Periode (dagen)", min_value=5, max_value=90, value=14)
        use_web_ev = c7.checkbox("Web (Tavily)", value=WEB_DEFAULT, disabled=not TAVILY_API_KEY, key="ev_web")
        if st.button("Genereer updates", type="primary"):
            results = []
            bar = st.progress(0.0)
            for i, tkr in enumerate(tickers):
                bar.progress(i / len(tickers), text=f"{tkr}…")
                try:
                    results.append(summarize_events_for_ticker(tkr, days=days, web_allowed=use_web_ev))
                except Exception as e:
                    results.append({"ticker": tkr, "content": f"### {tkr}\n\nMislukt: {type(e).__name__}: {e}"})
            bar.empty()
            st.session_state["events_out"] = {"days": days, "results": results}

        if ev := st.session_state.get("events_out"):
            for i, out in enumerate(ev["results"]):
                st.markdown(out["content"])
                render_sources(out.get("doc_legend") or [], out.get("web_legend") or [], f"ev_{i}")
                st.markdown("---")
            combined = "\n\n---\n\n".join(o["content"] for o in ev["results"])
            st.download_button(
                "Download alle updates (Markdown)", combined,
                file_name=f"{date.today().isoformat()}_events_{ev['days']}d.md", mime="text/markdown",
            )  # fmt: skip
