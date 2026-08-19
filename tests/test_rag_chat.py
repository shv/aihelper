from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.llm.base import RepairAdviceResult
from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.rag.models import DocumentChunk, SearchResult
from app.rag.service import GroundedRepairAdviceResult
from app.schemas import (
    RagAnswerStatus,
    RepairAdvice,
    RepairRisk,
    RiskLevel,
    TokenUsage,
)
from main import app, get_rag_service


@dataclass
class FakeRagService:
    result: GroundedRepairAdviceResult | None = None
    error: LLMError | None = None
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    async def get_repair_advice(
        self,
        message: str,
        *,
        category: str | None = None,
    ) -> GroundedRepairAdviceResult:
        self.calls.append((message, category))

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise AssertionError("Fake RAG service result is not configured")

        return self.result


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def make_result(
    *,
    sources: list[SearchResult] | None = None,
) -> GroundedRepairAdviceResult:
    advice_result = RepairAdviceResult(
        advice=RepairAdvice(
            summary="Перед плиткой подготовьте основание",
            clarifying_questions=["Какая поверхность основания?"],
            recommendations=[
                "Очистить основание",
                "Нанести грунтовку и гидроизоляцию",
            ],
            risks=[
                RepairRisk(
                    level=RiskLevel.MEDIUM,
                    description="Отслоение плитки",
                    mitigation="Соблюдать технологию подготовки",
                )
            ],
            requires_professional=False,
        ),
        model="fake-model",
        usage=TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        ),
    )
    return GroundedRepairAdviceResult(
        status=RagAnswerStatus.ANSWERED,
        advice=advice_result.advice,
        model=advice_result.model,
        usage=advice_result.usage,
        sources=[] if sources is None else sources,
    )


def make_insufficient_context_result() -> GroundedRepairAdviceResult:
    return GroundedRepairAdviceResult(
        status=RagAnswerStatus.INSUFFICIENT_CONTEXT,
        advice=RepairAdvice(
            summary="В базе знаний нет данных для ответа на этот вопрос.",
            clarifying_questions=[],
            recommendations=[],
            risks=[],
            requires_professional=False,
        ),
        model=None,
        usage=None,
        sources=[],
    )


def make_source() -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id="tile-waterproofing",
            title="Подготовка мокрой зоны",
            text="Основание очищают, грунтуют и гидроизолируют.",
            metadata={"category": "tile"},
        ),
        score=0.912345,
    )


def test_chat_rag_returns_grounded_advice_and_sources(
    client: TestClient,
) -> None:
    fake_service = FakeRagService(result=make_result(sources=[make_source()]))
    app.dependency_overrides[get_rag_service] = lambda: fake_service

    response = client.post(
        "/chat/rag",
        json={
            "message": "Что сделать перед укладкой плитки в ванной?",
            "category": "tile",
        },
    )

    assert response.status_code == 200
    assert fake_service.calls == [
        ("Что сделать перед укладкой плитки в ванной?", "tile")
    ]
    assert response.json() == {
        "status": "answered",
        "advice": {
            "summary": "Перед плиткой подготовьте основание",
            "clarifying_questions": ["Какая поверхность основания?"],
            "recommendations": [
                "Очистить основание",
                "Нанести грунтовку и гидроизоляцию",
            ],
            "risks": [
                {
                    "level": "medium",
                    "description": "Отслоение плитки",
                    "mitigation": "Соблюдать технологию подготовки",
                }
            ],
            "requires_professional": False,
        },
        "model": "fake-model",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        },
        "sources": [
            {
                "id": "tile-waterproofing",
                "title": "Подготовка мокрой зоны",
                "text": "Основание очищают, грунтуют и гидроизолируют.",
                "score": 0.912345,
            }
        ],
    }


def test_chat_rag_passes_none_when_category_is_omitted(
    client: TestClient,
) -> None:
    fake_service = FakeRagService(result=make_result())
    app.dependency_overrides[get_rag_service] = lambda: fake_service

    response = client.post(
        "/chat/rag",
        json={"message": "Как подготовить основание?"},
    )

    assert response.status_code == 200
    assert fake_service.calls == [("Как подготовить основание?", None)]
    assert response.json()["sources"] == []


def test_chat_rag_returns_explicit_insufficient_context_response(
    client: TestClient,
) -> None:
    fake_service = FakeRagService(result=make_insufficient_context_result())
    app.dependency_overrides[get_rag_service] = lambda: fake_service

    response = client.post(
        "/chat/rag",
        json={"message": "Как настроить Wi-Fi роутер?"},
    )

    assert response.status_code == 200
    assert fake_service.calls == [("Как настроить Wi-Fi роутер?", None)]
    assert response.json() == {
        "status": "insufficient_context",
        "advice": {
            "summary": "В базе знаний нет данных для ответа на этот вопрос.",
            "clarifying_questions": [],
            "recommendations": [],
            "risks": [],
            "requires_professional": False,
        },
        "model": None,
        "usage": None,
        "sources": [],
    }


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (LLMTimeoutError(), 504, "LLM provider timed out"),
        (
            LLMRateLimitError(),
            503,
            "LLM provider is temporarily overloaded",
        ),
        (LLMUnavailableError(), 503, "LLM provider is unavailable"),
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
def test_chat_rag_maps_llm_errors_to_http(
    client: TestClient,
    error: LLMError,
    expected_status: int,
    expected_detail: str,
) -> None:
    fake_service = FakeRagService(error=error)
    app.dependency_overrides[get_rag_service] = lambda: fake_service

    response = client.post(
        "/chat/rag",
        json={"message": "Тестовый вопрос"},
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "x" * 4_001},
        {},
    ],
    ids=["empty-message", "too-long-message", "missing-message"],
)
def test_chat_rag_rejects_invalid_request(
    client: TestClient,
    payload: dict[str, str],
) -> None:
    fake_service = FakeRagService(result=make_result())
    app.dependency_overrides[get_rag_service] = lambda: fake_service

    response = client.post("/chat/rag", json=payload)

    assert response.status_code == 422
    assert fake_service.calls == []
