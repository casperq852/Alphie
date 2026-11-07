"""Shared date parsing, name normalisation and score shaping for both vector backends."""

import math
import os
import re
from datetime import date, datetime, timedelta

import dateparser

RECENCY_TAU_DAYS = float(os.getenv("RECENCY_TAU_DAYS", "120"))  # e-folding time of the recency decay

_MONTHS = (
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|"
    r"september|october|november|december)"
)
DATE_PATTERNS = [
    r"\b\d{4}-\d{1,2}-\d{1,2}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{1,2}\s+" + _MONTHS + r"\s+\d{4}\b",
    r"\b" + _MONTHS + r"\s+\d{1,2},\s*\d{4}\b",
]
_DATE_RES = [re.compile(p, re.IGNORECASE) for p in DATE_PATTERNS]


def to_date(v) -> date | None:
    if not v:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return datetime.fromisoformat(str(v).strip()[:19]).date()
    except Exception:
        return None


def dates_in_text(text: str, past_only: bool = True) -> list[str]:
    """All dates found in text as ISO strings, newest first. Future dates (forecasts) are dropped by default."""
    if not text:
        return []
    today = date.today()
    out = set()
    for rx in _DATE_RES:
        for m in rx.finditer(text):
            d = dateparser.parse(m.group(0))
            if d and (not past_only or d.date() <= today):
                out.add(d.date().isoformat())
    return sorted(out, reverse=True)


def date_from_text(text: str, prefer_recent: bool = True) -> str | None:
    found = dates_in_text(text)
    if not found:
        return None
    return found[0] if prefer_recent else found[-1]


_REL_PERIODS = {
    # English + Dutch phrasings -> days back (0 = single day)
    "today": 0, "vandaag": 0,
    "last week": 7, "past week": 7, "afgelopen week": 7, "vorige week": 7,
    "last month": 30, "past month": 30, "afgelopen maand": 30, "vorige maand": 30,
    "last quarter": 92, "past quarter": 92, "afgelopen kwartaal": 92, "vorig kwartaal": 92,
    "last year": 365, "past year": 365, "afgelopen jaar": 365, "vorig jaar": 365,
}  # fmt: skip
_REL_N_RE = re.compile(r"\b(?:last|past|laatste|afgelopen)\s+(\d{1,3})\s+(days?|dagen|weeks?|weken)\b", re.I)


def parse_query_date_focus(q: str) -> tuple[str | None, str | None, str | None]:
    """Return (exact_date, date_from, date_to) implied by the query text."""
    q_low = (q or "").lower()
    today = date.today()
    if "yesterday" in q_low or "gisteren" in q_low:
        return (today - timedelta(days=1)).isoformat(), None, None
    m = _REL_N_RE.search(q_low)
    if m:
        n = int(m.group(1)) * (7 if m.group(2).startswith("we") else 1)
        return None, (today - timedelta(days=n)).isoformat(), today.isoformat()
    for phrase, days in _REL_PERIODS.items():
        if phrase in q_low:
            if days == 0:
                return today.isoformat(), None, None
            return None, (today - timedelta(days=days)).isoformat(), today.isoformat()
    found = dates_in_text(q, past_only=False)
    if len(found) == 1:
        return found[0], None, None
    if len(found) >= 2:
        return None, min(found), max(found)
    return None, None, None


def norm_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(group|groep|plc|inc|nv|n v|sa|ltd|corp|corporation|co|ag|se|holding|holdings|spa|asa|ab)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def names_match(a: str, b: str) -> bool:
    """Whole-word containment either way on normalised names ("asml" ~ "asml holding nv")."""
    na, nb = norm_name(a), norm_name(b)
    if not na or not nb:
        return False
    return f" {na} " in f" {nb} " or f" {nb} " in f" {na} "


def adjust_score(
    score: float,
    chunk_date=None,
    doc_date=None,
    added_at=None,
    date_focus: tuple[str | None, str | None, str | None] = (None, None, None),
    recency_tau_days: float | None = None,
) -> float:
    """Recency decay + bonus for chunks matching a date mentioned in the query."""
    tau = RECENCY_TAU_DAYS if recency_tau_days is None else recency_tau_days
    today = date.today()
    base = to_date(chunk_date) or to_date(doc_date) or to_date(added_at)
    adj = float(score)
    if base and tau > 0:
        adj *= math.exp(-max(0, (today - base).days) / tau)

    exact, d_from, d_to = date_focus
    cd = to_date(chunk_date) or to_date(doc_date)
    if exact:
        ex = to_date(exact)
        if cd and ex:
            delta = abs((cd - ex).days)
            adj *= math.exp(-(delta**2) / (2 * 2.0**2)) * 1.5
        else:
            adj *= 0.85
    elif d_from and d_to:
        f, t = to_date(d_from), to_date(d_to)
        if cd and f and t and f <= cd <= t:
            adj *= 1.4
        else:
            adj *= 0.9
    return adj
