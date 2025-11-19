import os
import requests
from typing import List, Dict, Tuple

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

import re
from datetime import datetime

def _norm_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(group|plc|inc|nv|sa|ltd|corp|co|ag|se|nv)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _looks_recent_snippet(text: str, max_age_days: int = 120) -> float:
    """Heuristic: reward current-year/month mentions; penalize likely old snippets."""
    now = datetime.now()
    score = 1.0
    # Bonus if current year appears
    if str(now.year) in (text or ""):
        score *= 1.2
    # Mild penalty if only older years appear
    older_hits = re.findall(r"\b(20\d{2})\b", text or "")
    if older_hits and all(int(y) <= now.year - 1 for y in older_hits):
        score *= 0.8
    return score

def _company_match_ok(snippet: dict, companies: list[str]) -> bool:
    if not companies: 
        return True
    title = (snippet.get("title") or "")
    content = (snippet.get("content") or "")
    blob = f"{title}\n{content}".lower()
    # must contain at least one normalized company token
    for c in companies:
        n = _norm_name(c)
        if not n: 
            continue
        # require all tokens of the normalized name to appear
        tokens = [t for t in n.split() if len(t) > 1]
        if tokens and all(t in blob for t in tokens):
            return True
    return False

def _post_json(url: str, payload: dict, timeout: int = 25) -> Tuple[dict, str]:
    """POST JSON and return (json_or_empty, error_msg_or_empty)."""
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), ""
    except Exception as e:
        return {}, f"{type(e).__name__}: {e}"

def tavily_search(query: str, max_results: int = 5) -> Tuple[List[Dict], str]:
    """
    Primary search on /search; returns (results, error_message_if_any).
    Results schema: [{title, url, content}]
    """
    if not TAVILY_API_KEY:
        return [], "No TAVILY_API_KEY set"

    base_payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "include_answer": False,
        "search_depth": "advanced",
        "include_domains": [],
        "exclude_domains": [],
    }
    try:
        r = requests.post("https://api.tavily.com/search", json=base_payload, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"
    results = []
    for r in data.get("results", [])[:max_results]:
        results.append({
            "title": (r.get("title") or "").strip(),
            "url": (r.get("url") or "").strip(),
            "content": (r.get("content") or "")[:1200]
        })
    return results, ""

def tavily_news(query: str, max_results: int = 5, time_range: str = "week") -> Tuple[List[Dict], str]:
    """
    Fallback to /news; returns (results, error_message_if_any).
    """
    if not TAVILY_API_KEY:
        return [], "No TAVILY_API_KEY set"

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "time_range": time_range,  # day|week|month
    }
    try:
        r = requests.post("https://api.tavily.com/news", json=payload, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"
    results = []
    for r in data.get("results", [])[:max_results]:
        results.append({
            "title": (r.get("title") or "").strip(),
            "url": (r.get("url") or "").strip(),
            "content": (r.get("content") or "")[:1200]
        })
    return results, ""

def web_augmented_snippets(user_question: str, companies: List[str], per_company: int = 2, max_total: int = 6) -> Tuple[List[Dict], List[str]]:
    if not TAVILY_API_KEY:
        return [], ["No TAVILY_API_KEY set"]

    snippets, debug = [], []
    queries_ran = 0

    def _keep(hits):
        out = []
        for h in hits:
            if not _company_match_ok(h, companies):
                continue
            h["_score"] = _looks_recent_snippet(h.get("content",""))
            out.append(h)
        # order by our recency score then keep original order
        out.sort(key=lambda x: x.get("_score", 1.0), reverse=True)
        return out

    # Broad (only if we truly have no companies)
    if not companies:
        queries_ran += 1
        q = f"{user_question}. Prefer earnings, guidance, regulatory actions, M&A, or broker calls."
        hits, err1 = tavily_search(q, max_results=max_total)
        if err1: debug.append(f"/search error (broad): {err1}")
        if not hits:
            news_hits, err2 = tavily_news(q, max_results=max_total, time_range="month")
            if err2: debug.append(f"/news error (broad): {err2}")
            hits = news_hits
        snippets.extend(_keep(hits))

    # Targeted per company
    for c in companies[:4]:
        if len(snippets) >= max_total: break
        q = (f"{c} earnings, guidance, regulatory updates, M&A, broker notes. "
             f"User focus: {user_question}")
        queries_ran += 1

        hits, err1 = tavily_search(q, max_results=per_company)
        if err1: debug.append(f"/search error for '{c}': {err1}")
        if not hits:
            news_hits, err2 = tavily_news(q, max_results=per_company, time_range="month")
            if err2: debug.append(f"/news error for '{c}': {err2}")
            hits = news_hits

        snippets.extend(_keep(hits))

    if queries_ran == 0:
        debug.append("No queries ran (empty company list).")

    # final trim
    return snippets[:max_total], debug
