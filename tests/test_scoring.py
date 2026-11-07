from datetime import date, timedelta

from scoring import adjust_score, dates_in_text, names_match, parse_query_date_focus


def test_dutch_relative_periods():
    today = date.today()
    _, f, t = parse_query_date_focus("Wat gebeurde er de laatste 14 dagen bij ASML?")
    assert f == (today - timedelta(days=14)).isoformat() and t == today.isoformat()
    _, f, _ = parse_query_date_focus("nieuws van vorige week")
    assert f == (today - timedelta(days=7)).isoformat()


def test_explicit_dates_in_query():
    exact, f, t = parse_query_date_focus("results on 2025-03-01")
    assert exact == "2025-03-01" and f is None and t is None


def test_future_dates_dropped():
    future = (date.today() + timedelta(days=400)).isoformat()
    assert dates_in_text(f"Report 2024-05-02, target by {future}") == ["2024-05-02"]


def test_names_match():
    assert names_match("ASML", "ASML Holding NV")
    assert names_match("JD Sports Fashion plc", "JD Sports Fashion")
    assert not names_match("ING", "Boeing")


def test_recency_decay():
    old = (date.today() - timedelta(days=365)).isoformat()
    new = date.today().isoformat()
    assert adjust_score(1.0, chunk_date=new) > adjust_score(1.0, chunk_date=old)
    assert adjust_score(1.0, chunk_date=old, recency_tau_days=0) == 1.0
