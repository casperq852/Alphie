"""Free-form 'custom query' mode with full control over retrieval parameters."""

from typing import Any

from agent_policy import detect_companies, format_pdf_context, format_web_context
from llm import chat
from rag import DETAIL_PRESETS, collect_evidence
from websearch import web_augmented_snippets

_SYSTEM_PROMPT = """
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
- Prefer the most recent evidence for figures (each source shows its date).
- Keep the tone professional and write in Dutch.

Avoid boilerplate. Always try to go a bit long momentum but don't mention explicitly. Avoid imposed section headers.
Always answer in Dutch. Don't refer to the user explicitly, always bring things factually. Don't mention PDF or
document specifically - distill the information and make your own arguments out of it.
""".strip()


def _custom_llm_answer(
    query: str, focus: str, pdf_blocks: list[str], web_blocks: list[str], model: str, max_tokens: int
) -> str:
    user_prompt = f"""
## USER QUERY
{query}

## USER-SPECIFIED FOCUS
{focus or "(none)"}

## PDF CONTEXT (primary evidence)
{chr(10).join(pdf_blocks) if pdf_blocks else "(no PDF context)"}

## WEB CONTEXT (optional)
{chr(10).join(web_blocks) if web_blocks else "(no web context)"}

## TASK
Provide the best possible answer using the evidence above.
If evidence is insufficient for any part of the user's request, explicitly say what cannot be answered.
""".strip()
    return chat(
        [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        model=model,
        max_tokens=max_tokens,
    )


def run_custom_query(
    focus_notes: str,
    user_query: str,
    use_web: bool,
    model: str,
    k_chunks: int = 18,
    min_date: str | None = None,
    detail: str = "elaborate",
    max_tokens: int = 2000,
    company_override: list[str] | None = None,
) -> dict[str, Any]:
    if not (user_query or "").strip():
        raise ValueError("Lege vraag.")
    merged_query = f"{user_query}\n\nUser-specified focus parameters:\n{focus_notes}" if focus_notes else user_query

    companies = company_override if company_override is not None else detect_companies(user_query)
    per_doc_cap = DETAIL_PRESETS.get(detail, DETAIL_PRESETS["elaborate"])["per_doc_cap"]

    hits = collect_evidence(
        merged_query, k=k_chunks, per_doc_cap=per_doc_cap, company_filter=companies or None, date_from=min_date
    )
    pdf_blocks, pdf_legend = format_pdf_context(hits)

    if use_web:
        web_snips, web_debug = web_augmented_snippets(user_query, companies)
    else:
        web_snips, web_debug = [], ["Web disabled"]
    web_blocks, web_legend = format_web_context(web_snips)

    answer = _custom_llm_answer(user_query, focus_notes, pdf_blocks, web_blocks, model, max_tokens)
    return {
        "content": answer, "legend": pdf_legend, "web_legend": web_legend, "companies": companies,
        "local_hits": len(hits), "web_debug": web_debug,
    }  # fmt: skip
