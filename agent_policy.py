"""Agents: event summaries per ticker and the multi-section investment case builder."""

from collections.abc import Callable
from typing import Any

from company_normalizer import canonicalize, classify_companies
from llm import OPENAI_MODEL, chat
from rag import collect_evidence, days_ago_iso, refresh_company_index, search_chunks
from websearch import web_augmented_snippets

Legend = list[tuple[str, str, str]]


def format_pdf_context(hits: list[dict]) -> tuple[list[str], Legend]:
    """[S#] blocks with source name + date (so the model can weigh recency) and the matching legend."""
    blocks, legend = [], []
    for i, h in enumerate(hits, start=1):
        tag = f"S{i}"
        title = h.get("source") or "PDF-bron"
        legend.append((tag, h.get("stored_path") or "", title))
        d = h.get("doc_date") or h.get("chunk_date") or ""
        header = f"[{tag}] {title}" + (f" ({d})" if d else "")
        blocks.append(f"{header}\n{h.get('text') or ''}")
    return blocks, legend


def format_web_context(snips: list[dict]) -> tuple[list[str], Legend]:
    blocks, legend = [], []
    for j, r in enumerate(snips, start=1):
        tag = f"W{j}"
        legend.append((tag, r.get("title", "web"), r.get("url", "")))
        d = r.get("published_date") or ""
        blocks.append(f"[{tag}]" + (f" ({d})" if d else "") + f" {r.get('snippet') or r.get('content') or ''}")
    return blocks, legend


def detect_companies(text: str) -> list[str]:
    """Companies in a free-text query, aligned to names known in the repository."""
    refresh_company_index()
    try:
        found = classify_companies(text)
    except Exception:
        found = []
    return canonicalize(found or [text.strip()])


def summarize_events_for_ticker(
    ticker: str,
    days: int = 14,
    web_allowed: bool = True,
    model_override: str | None = None,
) -> dict[str, Any]:
    """Samenvatting van key events voor één aandeel over de laatste N dagen.

    - PDF / Qdrant documenten (gepubliceerd binnen de periode) zijn **leidend**.
    - Web (Tavily nieuws binnen de periode) is secundair; alleen gebruiken als het echt iets toevoegt."""
    refresh_company_index()
    ticker = (canonicalize([ticker]) or [ticker])[0]

    # ---------- 1) PDF / document search (leading), restricted to the period ----------
    base_question = (
        f"Key bedrijfsspecifieke gebeurtenissen voor {ticker} in de laatste {days} dagen: "
        f"earnings, trading updates, guidance, grote koersbewegingen, "
        f"broker rating/target wijzigingen, M&A, strategische aankondigingen en relevante regulatoire nieuws."
    )
    local_hits = search_chunks(query=base_question, k=24, company_filter=[ticker], date_from=days_ago_iso(days))
    doc_blocks, doc_legend = format_pdf_context(local_hits)
    doc_context = "\n\n".join(doc_blocks)

    # ---------- 2) Web snippets (secondary / optional) ----------
    web_snips: list[dict] = []
    web_debug: list[str] = []
    if web_allowed:
        web_snips, web_debug = web_augmented_snippets(user_question=base_question, companies=[ticker], days=days)
    web_blocks, web_legend = format_web_context(web_snips)
    web_context = "\n\n".join(web_blocks)

    # ---------- 3) Geen evidence -> "rustige periode" ----------
    if not doc_context and not web_snips:
        content = (
            f"### {ticker}\n\n"
            f"**Samenvatting laatste {days} dagen**\n\n"
            f"- Er is in de afgelopen {days} dagen weinig tot geen materiële, bedrijfsspecifieke nieuwsflow "
            f"beschikbaar over {ticker} in de documenten of webbronnen.\n\n"
            f"**Impact op moat & langetermijnpositie**\n\n"
            f"- Geen duidelijke verandering in competitieve positie of moat op basis van de recente informatie.\n"
        )
        return {
            "ticker": ticker, "content": content, "doc_legend": doc_legend, "web_legend": web_legend,
            "web_debug": web_debug, "used_web": False,
        }  # fmt: skip

    # ---------- 4) Prompt: langere samenvatting, kortere moat ----------
    prompt = f"""
Je bent een fundamentele aandelenanalist. Schrijf in het Nederlands.

Je hebt twee soorten bronnen:

1. **PDF-bronnen [S#]** — interne researchdocumenten; deze zijn **leidend**.
2. **Web-bronnen [W#]** — optioneel; alleen gebruiken als ze echt recente, concrete informatie toevoegen.

Voor het aandeel **{ticker}**:

Begin met de kop `### {ticker}`.

**Samenvatting**
- Schrijf in 2–3 korte alinea’s.
- Vermijd openingszinnen als "In de afgelopen {days} dagen"; begin direct met de inhoud.
- Beschrijf de belangrijkste ontwikkelingen, bijvoorbeeld:
  * wijzigingen in guidance, winstverwachtingen of outlook,
  * koersreacties en hun aanleidingen,
  * relevante uitspraken van management,
  * deals, M&A of regulatoire gebeurtenissen.
- Gebruik **relatieve cijfers**, niet absolute. Dus zeg: “groei 3% tegenover verwachting 4%” of “marge daalde licht
  t.o.v. vorig kwartaal”, in plaats van kale bedragen of marges.
- Noem waar mogelijk richting en context: wat was verwacht, en hoe week het resultaat daarvan af?
- Geef een logische samenhang tussen nieuws, cijfers en koersreactie. Geef explicite nummers van koersreacties waar
  mogelijk (daling van x%, stijging van x% etc.)
- Wees streng en licht cynisch. Documenten en nieuws van bedrijven krijgen vaak een positieve draai; probeer de echte
  impact te duiden.
- Verwijs naar bronnen met [S#] en [W#]. Verzin geen cijfers die niet in de bronnen staan.

**Impact op moat & langetermijnpositie**
- Maximaal drie bullets, kort en to the point.
- Wees best streng. Als er negatief momentum is zal er meestal echt wel druk op de moat staan. Noem momentum niet
  expliciet maar redeneer wel in die richting.
- Geef per punt aan of de moat wordt **versterkt, verzwakt** of **grotendeels gelijk** blijft.
- Combineer verwante thema’s (pricing power + merk bijvoorbeeld).
- Richt je op de structurele impact: verandert dit iets fundamenteels aan competitieve positie, kostenvoordeel of
  merksterkte?

Gebruik vooral de PDF-context [S#]; gebruik de web-context [W#] alleen als die iets toevoegt.
Vermijd overbodige inleidingen of clichés.

PDF-context:
{doc_context or "(geen relevante PDF-fragmenten gevonden)"}

Web-context (optioneel):
{web_context or "(geen webfragmenten gebruikt of alleen ruis)"}
""".strip()

    content = chat([{"role": "user", "content": prompt}], model=model_override or OPENAI_MODEL, max_tokens=900)
    return {
        "ticker": ticker, "content": content, "doc_legend": doc_legend, "web_legend": web_legend,
        "web_debug": web_debug, "used_web": bool(web_snips),
    }  # fmt: skip


_CASE_SECTIONS: list[tuple[str, int]] = [
    (
        """
**Sectie 1. Samenvatting investment case**

- Begin de sectie met de markdown-kop: `## 1. Samenvatting investment case` op de eerste regel.
- Daarna 1–3 alinea's tekst.
- Vat samen: wat voor bedrijf dit is, kern van de strategie, belangrijkste bronnen van waardecreatie,
  kwaliteit en aard van de moat, risicoprofiel en jouw impliciete positioneringsadvies
  (bijv. defensief, cyclisch, compounder; aantrekkelijk/neutraal/minder aantrekkelijk).
- Geen koersdoel, wel kwalitatieve waarderingsinschatting (bijv. "aantrekkelijk geprijsd gezien RoTE en risico-profiel").
""",
        600,
    ),
    (
        """
**Sectie 2. Historie / Track record**

- Begin de sectie met de markdown-kop: `## 2. Historie en track record` op de eerste regel.
- Schets een chronologische geschiedenis op hoofdlijnen: ontstaan, grote strategische moves,
  mislukkingen, restructurings, crisis-episodes (indien relevant), en hoe het huidige profiel is ontstaan.
- Evalueer het track record: welke strategische keuzes waren goed / slecht? Wat zegt dit over de cultuur?
- Gebruik bij voorkeur concrete cijfers (ROIC, groei, impairments, YoY groei en dergelijken) waar beschikbaar met
  [S#]/[W#].
""",
        900,
    ),
    (
        """
**Sectie 3. Pijlers voor waardecreatie**

- Begin de sectie met de markdown-kop: `## 3. Pijlers voor waardecreatie` op de eerste regel.
- Schrijf daarna met duidelijke subkopjes 3a, 3b en 3c.

**3a. Is de industrie aantrekkelijk en heeft het bedrijf een gunstige positie in de value chain?**
- Beschrijf structuur van de industrie (groei, regulering, cycliciteit, margin-niveau).
- Plaats het bedrijf binnen de keten: wie zijn klanten, waar zit pricing power, hoe geconcentreerd is marktaandeel?
- Geef aan of dit een "moat-heavy" of meer commodity-omgeving is.

**3b. Welke concurrentievoordelen (moats) beschermen de winstgevendheid en met welke KPI’s monitoren we die?**
- Identificeer relevante moat-types uit o.a.:
  * Direct netwerk-effect / tweezijdig netwerk-effect
  * Brand / andere intangibles
  * Kennistechnologie / IP / patenten
  * Schaal- en kostenvoordelen (productiekosten, inkoop, distributie, automatisering)
  * Switching costs (financieel, psychologisch, contractueel, transactioneel)
  * Efficient scale / natuurlijke oligopolies
- Benoem alleen de 2–4 belangrijkste moats die echt gelden.
- Voor elke moat:
  * Leg kort uit waarom deze bestaat.
  * Koppel 1–3 concrete KPI’s die we kunnen volgen.

**3c. Wordt de onderneming goed gemanaged?**
- Beoordeel managementkwaliteit: track record, kapitaaltoewijzing, incentives, governance.
- Beschrijf capital return beleid (dividenden, buybacks).
- Leverage situatie? Zijn er schandalen of foutjes geweest?
- Benoem concrete doelstellingen van het management en in hoeverre die realistisch zijn.

Gebruik een verhalende stijl; bullets alleen waar nuttig.
""",
        1300,
    ),
    (
        """
**Sectie 4. Risico's – waar kunnen we het mishebben?**
- Begin de sectie met de markdown-kop: `## 4. Risico's – waar kunnen we het mishebben?` op de eerste regel.
- Focus op een beperkt aantal kernrisico’s (3–6) die echt materieel zijn.
- Benoem per risico:
  * Mechanisme: wat zou er precies fout gaan?
  * Impact: op welke KPI’s (bijv. groei, multiple, kosten) slaat dit neer?
  * Realisme: basiscase / tail / politiek / regulatoir / disruptie.
- Vermijd generieke onzin ("macro", "concurrentie") zonder uitwerking; maak het concreet.
""",
        700,
    ),
    (
        """
**Sectie 5. Conclusie**
- Begin de sectie met de markdown-kop: `## 5. Conclusie` op de eerste regel.
- Trek een duidelijke conclusie over de investment case:
  * Kwaliteit van het bedrijf (moat, management, balans).
  * Verwachte waardecreatie (rendement op kapitaal vs. kosten van kapitaal).
  * Waardering in grote lijnen (goedkoop / fair / duur vs. peers en eigen historie).
- Benoem expliciet waarom dit aandeel **wel of niet** aantrekkelijk is in de context van een gediversifieerde
  portefeuille.
- Sluit af met 3–5 korte zinnen die de kernthesis samenvatten.
""",
        650,
    ),
]


def agent_investment_case(
    user_query: str,
    company_filter: list[str] | None,
    web_allowed: bool,
    guidance: str = "",
    model_override: str | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Uitgebreide investment case in vijf secties (1 LLM-call per sectie, elk met de vorige als context):

    1. Samenvatting  2. Historie / track record  3. Pijlers voor waardecreatie  4. Risico's  5. Conclusie"""
    if not user_query.strip() and not company_filter:
        raise ValueError("Geef een bedrijfsnaam of vraag op.")

    def _tick(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    _tick("Bedrijf herkennen…", 0.02)
    refresh_company_index()
    companies = canonicalize(company_filter) if company_filter else detect_companies(user_query)
    company_label = companies[0] if companies else user_query.strip()

    base_question = (
        f"Investment case voor {company_label}: businessmodel, historisch track record, "
        f"strategische positie, concurrentievoordelen (moats), managementkwaliteit, "
        f"belangrijkste risico's en waardering."
    )

    # ---------- PDF search: diversified over documents, gentle recency decay (history matters here) ----------
    _tick("Research doorzoeken…", 0.08)
    local_hits = collect_evidence(
        base_question, k=28, per_doc_cap=6, company_filter=companies or None, recency_tau_days=365
    )
    ctx_blocks, legend = format_pdf_context(local_hits)
    context_text = "\n\n".join(ctx_blocks) if ctx_blocks else "(geen relevante PDF-context gevonden)"

    # ---------- Web ----------
    if web_allowed:
        _tick("Web doorzoeken…", 0.14)
        web_snips, web_debug = web_augmented_snippets(base_question, companies)
    else:
        web_snips, web_debug = [], ["Web disabled voor investment case"]
    wparts, web_legend = format_web_context(web_snips)
    web_text = "\n".join(wparts) if wparts else "(geen web-context of web uitgeschakeld)"

    pdf_tags = ", ".join(t for t, _, _ in legend) or "(geen)"
    web_tags = ", ".join(t for t, _, _ in web_legend) or "(geen)"
    guidance_block = f"\n\nAanvullende analisten-instructies:\n{guidance}\n" if guidance.strip() else ""
    no_pdf_note = (
        "\nLET OP: er zijn geen interne researchdocumenten over dit bedrijf gevonden. Benoem dat expliciet en wees "
        "terughoudend met harde cijfers.\n"
        if not local_hits
        else ""
    )

    shared_header = f"""
Je bent een fundamentele equity-analist. Schrijf in het **Nederlands**, helder en analytisch.

Gebruik primair de PDF-bronnen [{pdf_tags}] uit de interne researchdatabase; vul alleen aan met web-bronnen
[{web_tags}] als ze iets nieuws toevoegen.
Voor bedrijfsbeschrijving, historie en strategie mag je ook oudere bronnen gebruiken.
Voor harde cijfers (omzet, EBITDA, nettowinst, leverage, CET1, RoTE, etc.) geef je voorkeur aan informatie uit de
meest recente 2–3 jaar in de context (de datum staat achter elke bron).
Als de actualiteit van een cijfer onduidelijk is, gebruik dan bij benadering/kwalitatieve termen ("in recente jaren",
"rond X%") of benoem dat het een indicatieve orde van grootte is.
Verwijs waar zinvol naar bronnen met **[S#]** (PDF) en **[W#]** (web).
{no_pdf_note}
Bedrijf / focus: **{company_label}**
{guidance_block}
PDF-context:
{context_text}

Web-context (optioneel):
{web_text}
""".strip()

    model = model_override or OPENAI_MODEL
    sections: list[str] = []
    for n, (instruction, max_tokens) in enumerate(_CASE_SECTIONS, start=1):
        _tick(f"Sectie {n}/{len(_CASE_SECTIONS)} schrijven…", 0.2 + 0.8 * (n - 1) / len(_CASE_SECTIONS))
        prev = "\n\n".join(sections)
        prev_block = (
            f"\n\nReeds geschreven secties van dezelfde investment case (voor consistentie, niet herschrijven):\n{prev}\n"
            if prev
            else ""
        )
        prompt = f"""{shared_header}
{prev_block}

Schrijf nu **uitsluitend** de volgende sectie van de investment case.
Gebruik een goed onderbouwde, verhalende stijl met duidelijke redenering.
Schrijf geen inleiding over het format, alleen de gevraagde sectie.

{instruction}
""".strip()
        sections.append(chat([{"role": "user", "content": prompt}], model=model, max_tokens=max_tokens))
    _tick("Klaar", 1.0)

    pdf_source_line = (
        "PDF-bronnen: " + ", ".join(f"[{t}] {name}" for t, _, name in legend) if legend else "PDF-bronnen: (geen)"
    )
    web_source_line = (
        "Web-bronnen: " + ", ".join(f"[{t}] {title}" for t, title, _ in web_legend)
        if web_legend
        else "Web-bronnen: (geen of uitgeschakeld)"
    )
    content = "\n\n".join(sections) + f"\n\n---\n**Bronnen**\n\n{pdf_source_line}\n\n{web_source_line}\n"

    return {
        "content": content, "legend": legend, "web_legend": web_legend, "company": company_label,
        "companies": companies, "agent_used_web": bool(web_allowed), "agent_local_hits": len(local_hits),
        "agent_top_score": local_hits[0].get("score", 0.0) if local_hits else 0.0, "agent_web_debug": web_debug,
    }  # fmt: skip
