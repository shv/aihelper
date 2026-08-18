import json
from dataclasses import dataclass, field

import pytest

from app.llm.base import RepairAdviceResult
from app.rag.models import DocumentChunk, SearchResult
from app.rag.service import GroundedRepairAdviceResult, RagService
from app.schemas import RepairAdvice, TokenUsage

EMBEDDING_MODEL = "test-embedding-model"
QUERY_EMBEDDING = [0.1, 0.2, 0.3]


class FakeEmbeddingProvider:
    def __init__(self, embeddings: list[list[float]] | None = None) -> None:
        self._embeddings = embeddings if embeddings is not None else [QUERY_EMBEDDING]
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self._embeddings


@dataclass
class FakeSearchStore:
    results: list[SearchResult] = field(default_factory=list)
    calls: list[tuple[list[float], str, int, str | None]] = field(default_factory=list)

    async def search(
        self,
        query_embedding: list[float],
        *,
        embedding_model: str,
        top_k: int,
        category: str | None = None,
    ) -> list[SearchResult]:
        self.calls.append((query_embedding, embedding_model, top_k, category))
        return self.results


@dataclass
class FakeGroundedAdviceProvider:
    result: RepairAdviceResult
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def get_grounded_repair_advice(
        self,
        message: str,
        context: str,
    ) -> RepairAdviceResult:
        self.calls.append((message, context))
        return self.result


def make_advice_result() -> RepairAdviceResult:
    return RepairAdviceResult(
        advice=RepairAdvice(
            summary="Использовать гидроизоляцию",
            clarifying_questions=[],
            recommendations=["Подготовить основание"],
            risks=[],
            requires_professional=False,
        ),
        model="fake-model",
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        ),
    )


def make_search_result(
    chunk_id: str,
    *,
    title: str,
    text: str,
    score: float,
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            title=title,
            text=text,
            metadata={"category": "tile"},
        ),
        score=score,
    )


@pytest.mark.asyncio
async def test_get_repair_advice_orchestrates_complete_rag_pipeline() -> None:
    message = "Что сделать перед укладкой плитки в ванной?"
    search_results = [
        make_search_result(
            "tile-waterproofing",
            title="Подготовка мокрой зоны",
            text="Основание очищают, грунтуют и гидроизолируют.",
            score=0.91,
        ),
        make_search_result(
            "tile-adhesive",
            title="Выбор плиточного клея",
            text="Клей выбирают с учётом основания и формата плитки.",
            score=0.82,
        ),
    ]
    embedding_provider = FakeEmbeddingProvider()
    search_store = FakeSearchStore(results=search_results)
    advice_result = make_advice_result()
    advice_provider = FakeGroundedAdviceProvider(advice_result)
    service = RagService(
        embedding_provider,
        search_store,
        advice_provider,
        embedding_model=EMBEDDING_MODEL,
        top_k=2,
    )

    result = await service.get_repair_advice(message)

    assert result == GroundedRepairAdviceResult(
        advice_result=advice_result,
        sources=search_results,
    )
    assert embedding_provider.calls == [[message]]
    assert search_store.calls == [(QUERY_EMBEDDING, EMBEDDING_MODEL, 2, None)]
    assert len(advice_provider.calls) == 1
    called_message, serialized_context = advice_provider.calls[0]
    assert called_message == message
    assert json.loads(serialized_context) == [
        {
            "id": "tile-waterproofing",
            "title": "Подготовка мокрой зоны",
            "text": "Основание очищают, грунтуют и гидроизолируют.",
        },
        {
            "id": "tile-adhesive",
            "title": "Выбор плиточного клея",
            "text": "Клей выбирают с учётом основания и формата плитки.",
        },
    ]


@pytest.mark.asyncio
async def test_get_repair_advice_passes_category_to_search() -> None:
    embedding_provider = FakeEmbeddingProvider()
    search_store = FakeSearchStore()
    advice_provider = FakeGroundedAdviceProvider(make_advice_result())
    service = RagService(
        embedding_provider,
        search_store,
        advice_provider,
        embedding_model=EMBEDDING_MODEL,
        top_k=3,
    )

    await service.get_repair_advice("Как подготовить ванную?", category="tile")

    assert search_store.calls == [(QUERY_EMBEDDING, EMBEDDING_MODEL, 3, "tile")]


@pytest.mark.asyncio
async def test_empty_retrieval_still_calls_llm_with_empty_context() -> None:
    embedding_provider = FakeEmbeddingProvider()
    search_store = FakeSearchStore(results=[])
    advice_provider = FakeGroundedAdviceProvider(make_advice_result())
    service = RagService(
        embedding_provider,
        search_store,
        advice_provider,
        embedding_model=EMBEDDING_MODEL,
        top_k=3,
    )

    result = await service.get_repair_advice("Неизвестный вопрос")

    assert result.sources == []
    assert advice_provider.calls == [("Неизвестный вопрос", "[]")]


@pytest.mark.parametrize("top_k", [-1, 0])
def test_rejects_non_positive_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        RagService(
            FakeEmbeddingProvider(),
            FakeSearchStore(),
            FakeGroundedAdviceProvider(make_advice_result()),
            embedding_model=EMBEDDING_MODEL,
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "embeddings",
    [[], [QUERY_EMBEDDING, [0.4, 0.5, 0.6]]],
    ids=["missing", "extra"],
)
@pytest.mark.asyncio
async def test_rejects_wrong_number_of_query_embeddings(
    embeddings: list[list[float]],
) -> None:
    search_store = FakeSearchStore()
    advice_provider = FakeGroundedAdviceProvider(make_advice_result())
    service = RagService(
        FakeEmbeddingProvider(embeddings),
        search_store,
        advice_provider,
        embedding_model=EMBEDDING_MODEL,
        top_k=3,
    )

    with pytest.raises(ValueError):
        await service.get_repair_advice("Вопрос")

    assert search_store.calls == []
    assert advice_provider.calls == []
