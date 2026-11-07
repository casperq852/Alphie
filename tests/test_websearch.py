import websearch


class _Resp:
    def raise_for_status(self):
        pass

    def json(self):
        return {
            "results": [
                {"title": "ASML beats", "url": "https://x/1", "content": "ASML Q3 beat", "published_date": "x"},
                {"title": "Other", "url": "https://x/2", "content": "unrelated"},
            ]
        }


def test_tavily_uses_search_endpoint_with_bearer_and_news_topic(monkeypatch):
    calls = []
    monkeypatch.setattr(websearch, "TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(
        websearch.requests, "post", lambda url, json, headers, timeout: calls.append((url, json, headers)) or _Resp()
    )
    snips, debug = websearch.web_augmented_snippets("events", ["ASML"], days=14)
    url, payload, headers = calls[0]
    assert url.endswith("/search") and headers["Authorization"] == "Bearer tvly-test"
    assert payload["topic"] == "news" and payload["time_range"] == "month" and "api_key" not in payload
    assert [s["url"] for s in snips] == ["https://x/1"]  # company filter drops the unrelated hit
    assert not debug
