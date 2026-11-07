from llm import REASONING_TOKEN_BUFFER, build_chat_params, is_reasoning_model, strip_code_fences


def test_reasoning_model_detection():
    assert is_reasoning_model("gpt-5-mini")
    assert is_reasoning_model("o3-mini")
    assert not is_reasoning_model("gpt-4o-mini")


def test_reasoning_params_drop_temperature_and_use_completion_tokens():
    p = build_chat_params("gpt-5-mini", temperature=0.2, max_tokens=900)
    assert "temperature" not in p and "max_tokens" not in p
    assert p["max_completion_tokens"] == 900 + REASONING_TOKEN_BUFFER


def test_classic_params():
    assert build_chat_params("gpt-4o-mini", 0.2, 900) == {"model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 900}


def test_strip_code_fences():
    assert strip_code_fences('```json\n["ASML"]\n```') == '["ASML"]'
