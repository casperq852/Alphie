# custom_query.py

from typing import Optional, List, Dict, Any
from openai import OpenAI
from rag import search_chunks, classify_companies
from websearch import web_augmented_snippets
from company_normalizer import canonicalize, load_company_index
from rag import list_all_companies

client = OpenAI()


# ------------------------------------------------------------
#   NEW: clean, model-agnostic answer generator
# ------------------------------------------------------------
def _custom_llm_answer(
    query: str,
    focus: str,
    pdf_blocks: List[str],
    web_blocks: List[str],
    model: str,
    max_tokens: int,
) -> str:
    """
    Clean custom-query LLM logic.
    Works with gpt-4o-mini, gpt-5-mini, and avoids answer_question() entirely.
    No fixed structure. No hallucinated numbers. No token issues.
    """

    pdf_context = "\n\n".join(pdf_blocks) if pdf_blocks else "(no PDF context)"
    web_context = "\n\n".join(web_blocks) if web_blocks else "(no web context)"

    system_prompt = f"""
        You are a meticulous financial research analyst.

        Use the user's *focus instructions* as a guide - but some creativity is allowed.
        Use PDF evidence first, web evidence only if it adds real value.

        Rules:
        - Never invent numbers or quotes.
        - Never fabricate events.
        - If something is missing in the evidence, explicitly state that it is missing.
        - Synthesize information logically.
        - Provide a clear, analytical, defensible answer.
        - Use numbers and figures when they strengthen your point - this is generally the best way to strengthen your arguments.
        - Keep the tone professional and write in Dutch.

        Avoid boilerplate. Always try to go a bit long momentum but don't mention explicitly. Avoid imposed section headers. Always answer in Dutch. Don't refer to the user explicitly, always bring things factually. Don't mention PDF or document specifically - distill the information and make your own arguments out of it.
        """

    user_prompt = f"""
## USER QUERY
{query}

## USER-SPECIFIED FOCUS
{focus}

## PDF CONTEXT (primary evidence)
{pdf_context}

## WEB CONTEXT (optional)
{web_context}

## TASK
Provide the best possible answer using the evidence above.
If evidence is insufficient for any part of the user's request,
explicitly say what cannot be answered.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        max_completion_tokens=max_tokens,
    )

    return resp.choices[0].message.content



# ------------------------------------------------------------
#   MAIN FUNCTION: used by app.py
# ------------------------------------------------------------
def run_custom_query(
    focus_notes: str,
    user_query: str,
    use_web: bool,
    model: str,
    k_chunks: int = 18,
    min_date: Optional[str] = None,
    detail: str = "elaborate",
    max_tokens: int = 2000,
) -> Dict[str, Any]:
    """
    Ultimate flexible custom query function.
    Completely bypasses answer_question().
    """

    # Merge focus + user question
    merged_query = (
        f"{user_query}\n\n"
        f"User-specified focus parameters:\n{focus_notes}"
    )

    # ------------------------------------------------------------------
    # COMPANY DETECTION
    # ------------------------------------------------------------------
    # Load embedding index once (cheap)
    load_company_index(list_all_companies())

    raw_companies = classify_companies(merged_query, model=model)
    companies = canonicalize(raw_companies)

    # ------------------------------------------------------------------
    # PDF SEARCH
    # ------------------------------------------------------------------
    raw_hits = search_chunks(
        query=merged_query,
        k=k_chunks,
        company_filter=companies or None
    )

    # Optional min_date filtering
    if min_date:
        hits = [
            h for h in raw_hits
            if h.get("chunk_date") and h["chunk_date"] >= min_date
        ]
    else:
        hits = raw_hits

    # Build PDF context blocks
    pdf_blocks = []
    pdf_legend = []
    for i, h in enumerate(hits, start=1):
        tag = f"S{i}"
        pdf_legend.append((tag, h.get("stored_path", ""), h.get("source", "")))
        txt = h.get("text", "")
        pdf_blocks.append(f"[{tag}] {txt}")


    # ------------------------------------------------------------------
    # WEB SEARCH
    # ------------------------------------------------------------------
    if use_web:
        web_snips, web_debug = web_augmented_snippets(merged_query, companies)
    else:
        web_snips, web_debug = [], ["Web disabled"]

    web_blocks = []
    web_legend = []

    for j, r in enumerate(web_snips, start=1):
        tag = f"W{j}"
        web_legend.append((tag, r.get("title", "web"), r.get("url", "")))
        txt = r.get("snippet") or r.get("content") or ""
        web_blocks.append(f"[{tag}] {txt}")


    # ------------------------------------------------------------------
    # LLM ANSWER (new custom logic)
    # ------------------------------------------------------------------
    answer = _custom_llm_answer(
        query=user_query,
        focus=focus_notes,
        pdf_blocks=pdf_blocks,
        web_blocks=web_blocks,
        model=model,
        max_tokens=max_tokens,
    )


    # ------------------------------------------------------------------
    # RETURN STRUCTURED OUTPUT FOR UI
    # ------------------------------------------------------------------
    return {
        "content": answer,
        "legend": pdf_legend,
        "web_legend": web_legend,
        "companies": companies,
        "local_hits": len(hits),
        "raw_hits": len(raw_hits),
        "filtered_hits": len(hits),
        "web_debug": web_debug,
    }
