from typing import Dict, Any, List, Tuple, Optional

import os
from openai import OpenAI

from rag import search_chunks
from websearch import web_augmented_snippets
from company_normalizer import canonicalize, load_company_index
from rag import list_all_companies

# Use same default as the rest of the app if available
_OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_events_client = OpenAI()


def _needs_web(
    user_query: str,
    local_hits: List[dict],
    prefer_local: bool,
    web_allowed: bool,
) -> tuple[bool, str]:
    """
    Decide if we should call web search for normal Q&A mode.
    Very simple:
    - if web not allowed           -> False
    - if no local hits             -> True, "no-local-hits"
    - if prefer_local and hits     -> True, "blend-local-web"
    - otherwise                    -> True, "always-blend"
    """
    if not web_allowed:
        return False, "web-disabled"
    if not local_hits:
        return True, "no-local-hits"
    if prefer_local:
        return True, "blend-local-web"
    return True, "always-blend"

def summarize_events_for_ticker(
    ticker: str,
    days: int = 14,
    web_allowed: bool = True,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Samenvatting van key events voor één aandeel over de laatste N dagen.

    - PDF / Qdrant documenten zijn **leidend**.
    - Web (Tavily) is secundair; alleen gebruiken als het echt iets toevoegt.
    """

    load_company_index(list_all_companies())
    ticker = canonicalize([ticker])[0]


    # ---------- 1) PDF / document search (leading) ----------
    base_question = (
        f"Key bedrijfsspecifieke gebeurtenissen voor {ticker} in de laatste {days} dagen: "
        f"earnings, trading updates, guidance, grote koersbewegingen, "
        f"broker rating/target wijzigingen, M&A, strategische aankondigingen en relevante regulatoire nieuws."
    )
    chunk_query = (
        base_question
        + " Focus op ongeveer de laatste periode; als er weinig is gebeurd, is dat ook een geldige uitkomst."
    )

    # eerst: met company_filter
    local_hits = search_chunks(
        query=chunk_query,
        k=24,
        company_filter=[ticker],
    )

    doc_legend: List[Tuple[str, str, str]] = []
    doc_blocks: List[str] = []

    for i, h in enumerate(local_hits, start=1):
        tag = f"S{i}"
        title = h.get("source") or h.get("doc_title") or h.get("title") or "PDF-bron"
        path = h.get("stored_path") or ""
        txt = h.get("text") or h.get("chunk_text") or ""
        chunk_date = h.get("chunk_date") or h.get("doc_date") or ""

        doc_legend.append((tag, path, title))

        header = f"[{tag}] {title}"
        if chunk_date:
            header += f" ({chunk_date})"
        doc_blocks.append(f"{header}\n{txt}")

    doc_context = "\n\n".join(doc_blocks)

    # ---------- 2) Web snippets (secondary / optional) ----------
    web_snips: List[Dict[str, Any]] = []
    web_debug: List[str] = []
    web_legend: List[Tuple[str, str, str]] = []
    web_context = ""

    if web_allowed:
        web_snips, web_debug = web_augmented_snippets(
            user_question=base_question,
            companies=[ticker],
        )
        if web_snips:
            parts: List[str] = []
            for j, r in enumerate(web_snips, start=1):
                tag = f"W{j}"
                title = r.get("title", "web")
                url = r.get("url", "")
                snippet = r.get("snippet") or r.get("content") or ""
                web_legend.append((tag, title, url))
                parts.append(f"[{tag}] {snippet}")
            web_context = "\n\n".join(parts)

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
            "ticker": ticker,
            "content": content,
            "doc_legend": doc_legend,
            "web_legend": web_legend,
            "web_debug": web_debug,
            "used_web": False,
        }

    # ---------- 4) Prompt: langere Samenvatting, kortere moat ----------
    # ---------- 4) Prompt: langere Samenvatting, kortere moat ----------
    prompt = f"""
Je bent een fundamentele aandelenanalist. Schrijf in het Nederlands.

Je hebt twee soorten bronnen:

1. **PDF-bronnen [S#]** — interne researchdocumenten; deze zijn **leidend**.  
2. **Web-bronnen [W#]** — optioneel; alleen gebruiken als ze echt recente, concrete informatie toevoegen.

Voor het aandeel **{ticker}**:

**Samenvatting**
- Schrijf in 2–3 korte alinea’s.
- Vermijd openingszinnen als "In de afgelopen {days} dagen"; begin direct met de inhoud.
- Beschrijf de belangrijkste ontwikkelingen, bijvoorbeeld:
  * wijzigingen in guidance, winstverwachtingen of outlook,
  * koersreacties en hun aanleidingen,
  * relevante uitspraken van management,
  * deals, M&A of regulatoire gebeurtenissen.
- Gebruik **relatieve cijfers**, niet absolute. Dus zeg: “groei 3% tegenover verwachting 4%” of “marge daalde licht t.o.v. vorig kwartaal”, in plaats van kale bedragen of marges.
- Noem waar mogelijk richting en context: wat was verwacht, en hoe week het resultaat daarvan af?
- Geef een logische samenhang tussen nieuws, cijfers en koersreactie. Geef explicite nummers van koersreacties waar mogelijk (daling van x%, stijging van x% etc.)
- Wees streng en licht cynisch. Documenten en nieuws van bedrijven krijgen vaak een positieve draai; probeer de echte impact te duiden.

**Impact op moat & langetermijnpositie**
- Maximaal drie bullets, kort en to the point.
- Wees best streng. Als er negatief momentum is zal er meestal echt wel druk op de moat staan. Noem momentum niet expliciet maar redeneer wel in die richting.
- Geef per punt aan of de moat wordt **versterkt, verzwakt** of **grotendeels gelijk** blijft.
- Combineer verwante thema’s (pricing power + merk bijvoorbeeld).
- Richt je op de structurele impact: verandert dit iets fundamenteels aan competitieve positie, kostenvoordeel of merksterkte?

Gebruik vooral de PDF-context [S#]; gebruik de web-context [W#] alleen als die iets toevoegt.
Vermijd overbodige inleidingen of clichés.

PDF-context:
{doc_context or "(geen relevante PDF-fragmenten gevonden)"}

Web-context (optioneel):
{web_context or "(geen webfragmenten gebruikt of alleen ruis)"}
""".strip()

    model = model_override or _OPENAI_MODEL_DEFAULT
    resp = _events_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=900,
    )
    content = resp.choices[0].message.content

    return {
        "ticker": ticker,
        "content": content,
        "doc_legend": doc_legend,
        "web_legend": web_legend,
        "web_debug": web_debug,
        "used_web": bool(web_snips),
    }

def agent_investment_case(
    user_query: str,
    company_filter: Optional[List[str]],
    web_allowed: bool,
    guidance: str = "",                      # NEW
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bouwt een uitgebreide investment case volgens vast format:

    1. Samenvatting investment case
    2. Historie / Track record
    3. Pijlers voor waardecreatie
       3a. Industrie & positie in value chain
       3b. Concurrentievoordelen / moats + KPI's
       3c. Managementkwaliteit & capital allocation
    4. Risico's (waar kunnen we het mishebben?)
    5. Conclusie

    Werkt in meerdere LLM-calls (1 per blok) op basis van PDF-repository + optioneel web.
    """
    # ----- Canonicalize company names -----
    load_company_index(list_all_companies())
    if company_filter:
        company_filter = canonicalize(company_filter)
    else:
        # fallback: canonicalize user_query guesses later
        pass


    # ---------- 1) Basis: welke company / query gebruiken ----------
    effective_companies = company_filter or []

    # If no explicit filter was given, try to canonicalize the detected name
    if not effective_companies:
        auto = canonicalize([user_query.strip()])
        effective_companies = auto

    company_label = effective_companies[0]


    base_question = (
        f"Investment case voor {company_label}: businessmodel, historisch track record, "
        f"strategische positie, concurrentievoordelen (moats), managementkwaliteit, "
        f"belangrijkste risico's en waardering."
    )

    # ---------- 2) PDF / document search ----------
    local_hits = search_chunks(
        query=base_question,
        k=28,
        company_filter=effective_companies or None,
    )

    ctx_blocks: list[str] = []
    legend: list[tuple[str, str, str]] = []
    for i, h in enumerate(local_hits, start=1):
        tag = f"S{i}"
        legend.append((tag, h.get("stored_path", ""), h.get("source", "")))
        txt = h.get("text", "")
        ctx_blocks.append(f"[{tag}] {txt}")

    context_text = "\n\n".join(ctx_blocks) if ctx_blocks else "(geen relevante PDF-context gevonden)"

    # ---------- 3) Web snippets ----------
    if web_allowed:
        web_snips, web_debug = web_augmented_snippets(base_question, effective_companies)
    else:
        web_snips, web_debug = [], ["Web disabled voor investment case"]

    web_legend: list[tuple[str, str, str]] = []
    wparts: list[str] = []
    for j, r in enumerate(web_snips, start=1):
        tag = f"W{j}"
        web_legend.append((tag, r.get("title", "web"), r.get("url", "")))
        wparts.append(f"[{tag}] {r.get('snippet','') if 'snippet' in r else r.get('content','')}")
    web_text = "\n".join(wparts) if wparts else "(geen web-context of web uitgeschakeld)"

    # ---------- 4) Shared prompt header ----------
    pdf_tags = ", ".join(t for t, _, _ in legend) or "(geen)"
    web_tags = ", ".join(t for t, _, _ in web_legend) or "(geen)"

    guidance_block = f"\n\nAanvullende analisten-instructies:\n{guidance}\n" if guidance else ""

    shared_header = f"""
Je bent een fundamentele equity-analist. Schrijf in het **Nederlands**, helder en analytisch.

Gebruik primair de PDF-bronnen [{pdf_tags}] uit de interne researchdatabase; vul alleen aan met web-bronnen [{web_tags}] als ze iets nieuws toevoegen.
Voor bedrijfsbeschrijving, historie en strategie mag je ook oudere bronnen gebruiken.
Voor harde cijfers (omzet, EBITDA, nettowinst, leverage, CET1, RoTE, etc.) geef je voorkeur aan informatie uit de meest recente 2–3 jaar in de context.
Als de actualiteit van een cijfer onduidelijk is, gebruik dan bij benadering/kwalitatieve termen ("in recente jaren", "rond X%") of benoem dat het een indicatieve orde van grootte is.
Verwijs waar zinvol naar bronnen met **[S#]** (PDF) en **[W#]** (web).

Bedrijf / focus: **{company_label}**

{guidance_block}

PDF-context:
{context_text}

Web-context (optioneel):
{web_text}
""".strip()


    model = model_override or _OPENAI_MODEL_DEFAULT

    def _section(
        instruction: str,
        previous_sections: str = "",
        max_tokens: int = 750,
    ) -> str:
        """
        Eén sectie genereren. previous_sections wordt meegegeven zodat de case
        consistent blijft qua cijfers & conclusies.
        """
        prev_block = (
            ""
            if not previous_sections
            else f"\n\nReeds geschreven secties van dezelfde investment case (voor consistentie, niet herschrijven):\n{previous_sections}\n"
        )

        prompt = f"""{shared_header}

{prev_block}

Schrijf nu **uitsluitend** de volgende sectie van de investment case.
Gebruik een goed onderbouwde, verhalende stijl met duidelijke redenering.
Schrijf geen inleiding over het format, alleen de gevraagde sectie.

{instruction}
""".strip()

        resp = _events_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # ---------- 5) De vijf blokken, met aparte calls ----------
    sections: list[str] = []

    # 1. Samenvatting investment case
    sec1 = _section(
        """
**Sectie 1. Samenvatting investment case**

- Begin de sectie met de markdown-kop: `## 1. Samenvatting investment case` op de eerste regel.
- Daarna 1–3 alinea's tekst.
- Vat samen: wat voor bedrijf dit is, kern van de strategie, belangrijkste bronnen van waardecreatie,
  kwaliteit en aard van de moat, risicoprofiel en jouw impliciete positioneringsadvies
  (bijv. defensief, cyclisch, compounder; aantrekkelijk/neutraal/minder aantrekkelijk).
- Geen koersdoel, wel kwalitatieve waarderingsinschatting (bijv. "aantrekkelijk geprijsd gezien RoTE en risico-profiel").
""",
        previous_sections="",
        max_tokens=600,
    )
    sections.append(sec1)

    sec2 = _section(
        """
**Sectie 2. Historie / Track record**

- Begin de sectie met de markdown-kop: `## 2. Historie en track record` op de eerste regel.
- Schets een chronologische geschiedenis op hoofdlijnen: ontstaan, grote strategische moves,
  mislukkingen, restructurings, crisis-episodes (indien relevant), en hoe het huidige profiel is ontstaan.
- Evalueer het track record: welke strategische keuzes waren goed / slecht? Wat zegt dit over de cultuur?
- Gebruik bij voorkeur concrete cijfers (ROIC, groei, impairments, YoY groei en dergelijken) waar beschikbaar met [S#]/[W#].
""",
        previous_sections=sec1,
        max_tokens=900,
    )
    sections.append(sec2)

    # 3. Pijlers voor waardecreatie
    sec3 = _section(
        """
**Sectie 3. Pijlers voor waardecreatie**

- Begin de sectie met de markdown-kop: `## 3. Pijlers voor waardecreatie` op de eerste regel.
- Schrijf daarna met duidelijke subkopjes 3a, 3b en 3c.

**3a. Is de industrie aantrekkelijk en heeft het bedrijf een gunstige positie in de value chain?**
- Beschrijf structuur van de industrie (groei, regulering, cycliciteit, margin-niveau).
- Plaats het bedrijf binnen de keten: wie zijn klanten, waar zit pricing power, hoe concentrisch is marktaandeel?
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
  * Koppel 1–3 concrete KPI’s  die we kunnen volgen.

**3c. Wordt de onderneming goed gemanaged?**
- Beoordeel managementkwaliteit: track record, kapitaaltoewijzing, incentives, governance.
- Beschrijf capital return beleid (dividenden, buybacks).
- Leverage situatie? Zijn er schandalen of foutjes geweest?
- Benoem concrete doelstellingen van het management en in hoeverre die realistisch zijn.

Gebruik een verhalende stijl; bullets alleen waar nuttig.
""",
        previous_sections=sec1 + "\n\n" + sec2,
        max_tokens=1300,
    )
    sections.append(sec3)

    # 4. Risico's
    sec4 = _section(
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
        previous_sections=sec1 + "\n\n" + sec2 + "\n\n" + sec3,
        max_tokens=700,
    )
    sections.append(sec4)

    # 5. Conclusie
    sec5 = _section(
        """
**Sectie 5. Conclusie**
- Begin de sectie met de markdown-kop: `## 5. Conclusie` op de eerste regel.
- Trek een duidelijke conclusie over de investment case:
  * Kwaliteit van het bedrijf (moat, management, balans).
  * Verwachte waardecreatie (rendement op kapitaal vs. kosten van kapitaal).
  * Waardering in grote lijnen (goedkoop / fair / duur vs. peers en eigen historie).
- Benoem expliciet waarom dit aandeel **wel of niet** aantrekkelijk is in de context van een gediversifieerde portefeuille.
- Sluit af met 3–5 korte zinnen die de kernthesis samenvatten.
""",
        previous_sections=sec1 + "\n\n" + sec2 + "\n\n" + sec3 + "\n\n" + sec4,
        max_tokens=650,
    )
    sections.append(sec5)

    full_text = "\n\n".join(sections)

    # ---------- 6) Bronvermelding onderaan ----------
    pdf_source_line = (
        "PDF-bronnen: " + ", ".join(f"[{tag}]" for tag, _, _ in legend)
        if legend else "PDF-bronnen: (geen gevonden)"
    )
    web_source_line = (
        "Web-bronnen: " + ", ".join(f"[{tag}]" for tag, _, _ in web_legend)
        if web_legend else "Web-bronnen: (geen of uitgeschakeld)"
    )

    sources_block = f"\n\n---\n**Bronnen**\n\n{pdf_source_line}\n\n{web_source_line}\n"

    content = full_text + sources_block

    out: Dict[str, Any] = {
        "content": content,
        "legend": legend,
        "web_legend": web_legend,
    }
    out.update(
        {
            "agent_used_web": bool(web_allowed),
            "agent_web_reason": "investment-case" if web_allowed else "web-disabled",
            "agent_local_hits": len(local_hits),
            "agent_top_score": local_hits[0].get("score", 0.0) if local_hits else 0.0,
            "agent_web_debug": web_debug,
        }
    )
    return out

