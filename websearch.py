"""Optional web augmentation via Tavily (https://docs.tavily.com). Disabled when TAVILY_API_KEY is unset."""

import os
import re
from datetime import datetime

import requests

from scoring import norm_name

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"
_MAX_QUERY_CHARS = 380  # long prompts make poor search queries; Tavily works best with short queries


def _looks_recent_snippet(text: str) -> float:
    """Heuristic: reward current-year mentions; penalise snippets that only mention older years."""
    now = datetime.now()
    score = 1.0
    if str(now.year) in (text or ""):
        score *= 1.2
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text or "")]
    if years and all(y <= now.year - 1 for y in years):
        score *= 0.8
    return score


def _company_match_ok(snippet: dict, companies: list[str]) -> bool:
    """Snippet must mention every token of at least one (normalised) company name."""
    if not companies:
        return True
    blob = f"{snippet.get('title') or ''}\n{snippet.get('content') or ''}".lower()
    for c in companies:
        tokens = [t for t in norm_name(c).split() if len(t) > 1]
        if tokens and all(re.search(rf"\b{re.escape(t)}\b", blob) for t in tokens):
            return True
    return False


def _time_range(days: int | None) -> str | None:
    if not days:
        return None
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 31:
        return "month"
    return "year"


def tavily_search(
    query: str, max_results: int = 5, topic: str = "general", days: int | None = None
) -> tuple[list[dict], str]:
    """POST /search; returns (results, error_message). Results: [{title, url, content, published_date}]."""
    if not TAVILY_API_KEY:
        return [], "No TAVILY_API_KEY set"
    payload: dict = {
        "query": query[:_MAX_QUERY_CHARS],
        "max_results": max(1, min(20, max_results)),
        "search_depth": "advanced",
        "topic": topic,
        "include_answer": False,
    }
    tr = _time_range(days)
    if tr:
        payload["time_range"] = tr
    try:
        r = requests.post(TAVILY_URL, json=payload, headers={"Authorization": f"Bearer {TAVILY_API_KEY}"}, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"
    return [
        {
            "title": (x.get("title") or "").strip(),
            "url": (x.get("url") or "").strip(),
            "content": (x.get("content") or "")[:1200],
            "published_date": x.get("published_date") or "",
        }
        for x in data.get("results", [])[:max_results]
    ], ""


def web_augmented_snippets(
    user_question: str,
    companies: list[str],
    per_company: int = 3,
    max_total: int = 6,
    days: int | None = None,
) -> tuple[list[dict], list[str]]:
    """Company-targeted web snippets. `days` restricts to recent news (used by the events tab)."""
    if not TAVILY_API_KEY:
        return [], ["No TAVILY_API_KEY set"]

    snippets: list[dict] = []
    debug: list[str] = []
    seen_urls: set[str] = set()
    topic = "news" if days else "general"

    def _keep(hits: list[dict]) -> list[dict]:
        out = []
        for h in hits:
            if h["url"] in seen_urls or not _company_match_ok(h, companies):
                continue
            seen_urls.add(h["url"])
            h["_score"] = _looks_recent_snippet(f"{h.get('content', '')} {h.get('published_date', '')}")
            out.append(h)
        return sorted(out, key=lambda x: x["_score"], reverse=True)

    def _run(q: str, n: int, label: str) -> list[dict]:
        hits, err = tavily_search(q, max_results=n, topic=topic, days=days)
        if err:
            debug.append(f"search error ({label}): {err}")
        if not hits and topic == "general":  # fall back to the news index
            hits, err = tavily_search(q, max_results=n, topic="news", days=days or 30)
            if err:
                debug.append(f"news error ({label}): {err}")
        return hits

    if not companies:
        snippets.extend(_keep(_run(user_question, max_total, "broad")))
    for c in companies[:4]:
        if len(snippets) >= max_total:
            break
        q = f"{c} earnings guidance M&A regulatory broker rating. {user_question}"
        snippets.extend(_keep(_run(q, per_company, c)))
    return snippets[:max_total], debug
