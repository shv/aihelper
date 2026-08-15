from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from app.llm.base import RepairAdviceProvider, RepairAdviceResult
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.schemas import RepairAdvice, RepairRisk, RiskLevel, TokenUsage
from main import app, get_repair_advice_provider


@dataclass
class FakeRepairAdviceProvider:
    result: RepairAdviceResult | None = None
    error: LLMError | None = None

    async def get_repair_advice(self, message: str) -> RepairAdviceResult:
        if self.error is not None:
            raise self.error

        if self.result is None:
            raise AssertionError("Fake provider result is not configured")

        return self.result


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def make_result() -> RepairAdviceResult:
    return RepairAdviceResult(
        advice=RepairAdvice(
            summary="Тестовая рекомендация",
            clarifying_questions=["Какой размер помещения?"],
            recommendations=["Провести измерения"],
            risks=[
                RepairRisk(
                    level=RiskLevel.MEDIUM,
                    description="Ошибка измерения",
                    mitigation="Перепроверить размеры",
                )
            ],
            requires_professional=False,
        ),
        model="fake-model",
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        ),
    )


def test_chat_returns_structured_advice(client: TestClient) -> None:
    fake_provider: RepairAdviceProvider = FakeRepairAdviceProvider(result=make_result())

    app.dependency_overrides[get_repair_advice_provider] = lambda: fake_provider

    response = client.post(
        "/chat",
        json={"message": "Как положить плитку?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "advice": {
            "summary": "Тестовая рекомендация",
            "clarifying_questions": ["Какой размер помещения?"],
            "recommendations": ["Провести измерения"],
            "risks": [
                {
                    "level": "medium",
                    "description": "Ошибка измерения",
                    "mitigation": "Перепроверить размеры",
                }
            ],
            "requires_professional": False,
        },
        "model": "fake-model",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (
            LLMTimeoutError(),
            504,
            "LLM provider timed out",
        ),
        (
            LLMRateLimitError(),
            503,
            "LLM provider is temporarily overloaded",
        ),
        (
            LLMUnavailableError(),
            503,
            "LLM provider is unavailable",
        ),
        (
            LLMInvalidResponseError(),
            502,
            "LLM provider returned an invalid response",
        ),
        (
            LLMAuthenticationError(),
            500,
            "LLM provider is misconfigured",
        ),
    ],
)
def test_chat_maps_llm_errors_to_http(
    client: TestClient,
    error: LLMError,
    expected_status: int,
    expected_detail: str,
) -> None:
    fake_provider: RepairAdviceProvider = FakeRepairAdviceProvider(error=error)

    app.dependency_overrides[get_repair_advice_provider] = lambda: fake_provider

    response = client.post(
        "/chat",
        json={"message": "Тестовый вопрос"},
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


def test_chat_rejects_empty_message(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": ""},
    )

    assert response.status_code == 422
