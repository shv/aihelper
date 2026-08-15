import os

import pytest
from fastapi.testclient import TestClient

from main import app, get_openai_client

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        os.getenv("RUN_SMOKE_TESTS") != "1",
        reason="Real OpenAI smoke tests are disabled",
    ),
]


def test_real_chat_request() -> None:
    # Не использовать клиент, случайно закэшированный другими тестами.
    get_openai_client.cache_clear()
    app.dependency_overrides.clear()

    try:
        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={
                    "message": (
                        "Назови два уточняющих вопроса перед укладкой плитки в ванной."
                    )
                },
            )
    finally:
        get_openai_client.cache_clear()

    assert response.status_code == 200, response.text

    payload = response.json()

    assert payload["model"]
    assert payload["advice"]["summary"]
    assert isinstance(payload["advice"]["recommendations"], list)

    usage = payload["usage"]

    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["total_tokens"] == (usage["input_tokens"] + usage["output_tokens"])
