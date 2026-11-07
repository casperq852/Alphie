import company_normalizer as cn


def test_clean_company_list_keeps_real_companies():
    # These were all dropped by the old substring filters ("po", "iv", "ing", "sal", ...)
    names = [
        "JD Sports Fashion",
        "Givaudan",
        "Vivendi",
        "Boeing",
        "Booking Holdings",
        "Universal Music Group",
        "ING Groep",
        "Porsche",
        "Spotify",
        "Repsol",
    ]
    assert cn.clean_company_list(names, max_items=20) == names


def test_clean_company_list_drops_noise():
    noise = [
        "pembrolizumab",
        "10 mg tablet",
        "Bloomberg",
        "Kepler Cheuvreux",
        "UBS Global Research",
        "AZD-1234",
        "lowercase thing",
        "X",
    ]
    assert cn.clean_company_list(noise) == []


def test_clean_company_list_dedupes_case_insensitive():
    assert cn.clean_company_list(["ASML", "asml", "ASML"]) == ["ASML"]


def test_llm_canonicalize_falls_back_on_bad_output(monkeypatch):
    monkeypatch.setattr(cn, "chat", lambda *a, **k: "['not', 'json']")
    assert cn.llm_canonicalize(["Asml"]) == ["Asml"]
    monkeypatch.setattr(cn, "chat", lambda *a, **k: '```json\n["ASML Holding NV"]\n```')
    assert cn.llm_canonicalize(["Asml"]) == ["ASML Holding NV"]


def test_resolve_does_not_map_unknown_company(offline):
    cn.load_company_index(["ASML Holding", "Givaudan"])
    assert cn.resolve_by_embedding(["ASML Holding"]) == ["ASML Holding"]
    # unrelated name must stay as-is instead of being mapped onto the nearest known company
    assert cn.resolve_by_embedding(["Heineken"]) == ["Heineken"]
