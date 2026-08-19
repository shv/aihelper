import json
from dataclasses import dataclass, field

import pytest

from app.llm.base import RepairAdviceResult
from app.rag.models import DocumentChunk, SearchResult
from app.rag.service import GroundedRepairAdviceResult, RagService
from app.schemas import RagAnswerStatus, RepairAdvice, TokenUsage

EMBEDDING_MODEL = "test-embedding-model"
MIN_SCORE = 0.45
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
        min_score=MIN_SCORE,
    )

    result = await service.get_repair_advice(message)

    assert result == GroundedRepairAdviceResult(
        status=RagAnswerStatus.ANSWERED,
        advice=advice_result.advice,
        model=advice_result.model,
        usage=advice_result.usage,
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
        min_score=MIN_SCORE,
    )

    await service.get_repair_advice("Как подготовить ванную?", category="tile")

    assert search_store.calls == [(QUERY_EMBEDDING, EMBEDDING_MODEL, 3, "tile")]


@pytest.mark.asyncio
async def test_empty_retrieval_returns_deterministic_abstention_without_llm() -> None:
    embedding_provider = FakeEmbeddingProvider()
    search_store = FakeSearchStore(results=[])
    advice_provider = FakeGroundedAdviceProvider(make_advice_result())
    service = RagService(
        embedding_provider,
        search_store,
        advice_provider,
        embedding_model=EMBEDDING_MODEL,
        top_k=3,
        min_score=MIN_SCORE,
    )

    result = await service.get_repair_advice("Неизвестный вопрос")

    assert result == GroundedRepairAdviceResult(
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
    assert advice_provider.calls == []


@pytest.mark.parametrize("top_k", [-1, 0])
def test_rejects_non_positive_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        RagService(
            FakeEmbeddingProvider(),
            FakeSearchStore(),
            FakeGroundedAdviceProvider(make_advice_result()),
            embedding_model=EMBEDDING_MODEL,
            top_k=top_k,
            min_score=MIN_SCORE,
        )


@pytest.mark.parametrize("min_score", [-1.01, 1.01])
def test_rejects_min_score_outside_cosine_similarity_range(
    min_score: float,
) -> None:
    with pytest.raises(ValueError, match="min_score must be between -1 and 1"):
        RagService(
            FakeEmbeddingProvider(),
            FakeSearchStore(),
            FakeGroundedAdviceProvider(make_advice_result()),
            embedding_model=EMBEDDING_MODEL,
            top_k=3,
            min_score=min_score,
        )


@pytest.mark.asyncio
async def test_filters_results_below_min_score_from_context_and_sources() -> None:
    relevant_result = make_search_result(
        "gkl-cabinet",
        title="Крепление шкафа к ГКЛ",
        text="Шкаф крепят к стойкам или закладной.",
        score=0.75,
    )
    boundary_result = make_search_result(
        "tile-waterproofing",
        title="Гидроизоляция мокрой зоны",
        text="Перед плиткой выполняют гидроизоляцию.",
        score=MIN_SCORE,
    )
    weak_result = make_search_result(
        "laminate-gap",
        title="Зазор для ламината",
        text="У стены оставляют компенсационный зазор.",
        score=0.44,
    )
    advice_provider = FakeGroundedAdviceProvider(make_advice_result())
    service = RagService(
        FakeEmbeddingProvider(),
        FakeSearchStore(results=[relevant_result, boundary_result, weak_result]),
        advice_provider,
        embedding_model=EMBEDDING_MODEL,
        top_k=3,
        min_score=MIN_SCORE,
    )

    result = await service.get_repair_advice("Как повесить шкаф на ГКЛ?")

    assert result.sources == [relevant_result, boundary_result]
    assert len(advice_provider.calls) == 1
    _, serialized_context = advice_provider.calls[0]
    assert json.loads(serialized_context) == [
        {
            "id": "gkl-cabinet",
            "title": "Крепление шкафа к ГКЛ",
            "text": "Шкаф крепят к стойкам или закладной.",
        },
        {
            "id": "tile-waterproofing",
            "title": "Гидроизоляция мокрой зоны",
            "text": "Перед плиткой выполняют гидроизоляцию.",
        },
    ]


@pytest.mark.asyncio
async def test_returns_abstention_without_llm_when_all_results_are_below_min_score() -> (
    None
):
    weak_result = make_search_result(
        "laminate-gap",
        title="Зазор для ламината",
        text="У стены оставляют компенсационный зазор.",
        score=0.44,
    )
    advice_provider = FakeGroundedAdviceProvider(make_advice_result())
    service = RagService(
        FakeEmbeddingProvider(),
        FakeSearchStore(results=[weak_result]),
        advice_provider,
        embedding_model=EMBEDDING_MODEL,
        top_k=3,
        min_score=MIN_SCORE,
    )

    result = await service.get_repair_advice("Как настроить Wi-Fi роутер?")

    assert result.status is RagAnswerStatus.INSUFFICIENT_CONTEXT
    assert result.model is None
    assert result.usage is None
    assert result.sources == []
    assert advice_provider.calls == []


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
        min_score=MIN_SCORE,
    )

    with pytest.raises(ValueError):
        await service.get_repair_advice("Вопрос")

    assert search_store.calls == []
    assert advice_provider.calls == []
