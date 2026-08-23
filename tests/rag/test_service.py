import json
from dataclasses import dataclass, field

import pytest

from app.llm.base import RepairAdviceResult
from app.rag.models import DocumentChunk, SearchResult
from app.rag.service import GroundedRepairAdviceResult, RagService
from app.schemas import RagAnswerStatus, RepairAdvice, TokenUsage

EMBEDDING_MODEL = "test-embedding-model"
MIN_VECTOR_SCORE = 0.45
QUERY_EMBEDDING = [0.1, 0.2, 0.3]
VECTOR_CANDIDATE_TOP_K = 10
BM25_CANDIDATE_TOP_K = 8
CONTEXT_TOP_K = 3
RRF_K = 60


class FakeEmbeddingProvider:
    def __init__(self, embeddings: list[list[float]] | None = None) -> None:
        self._embeddings = embeddings if embeddings is not None else [QUERY_EMBEDDING]
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self._embeddings


@dataclass
class FakeSearchStore:
    vector_results: list[SearchResult] = field(default_factory=list)
    bm25_results: list[SearchResult] = field(default_factory=list)
    vector_calls: list[tuple[list[float], str, int, str | None]] = field(
        default_factory=list
    )
    bm25_calls: list[tuple[str, int, str | None]] = field(default_factory=list)

    async def search(
        self,
        query_embedding: list[float],
        *,
        embedding_model: str,
        top_k: int,
        category: str | None = None,
    ) -> list[SearchResult]:
        self.vector_calls.append((query_embedding, embedding_model, top_k, category))
        return self.vector_results

    async def search_bm25(
        self,
        query: str,
        *,
        top_k: int,
        category: str | None = None,
    ) -> list[SearchResult]:
        self.bm25_calls.append((query, top_k, category))
        return self.bm25_results


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
        usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
    )


def make_search_result(chunk_id: str, *, score: float) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            title=f"Title {chunk_id}",
            text=f"Text {chunk_id}",
            metadata={"category": "tile"},
        ),
        score=score,
    )


def make_service(
    embedding_provider: FakeEmbeddingProvider,
    search_store: FakeSearchStore,
    advice_provider: FakeGroundedAdviceProvider,
    *,
    vector_candidate_top_k: int,
    bm25_candidate_top_k: int,
    context_top_k: int,
    rrf_k: int,
    min_vector_score: float,
) -> RagService:
    return RagService(
        embedding_provider=embedding_provider,
        search_store=search_store,
        advice_provider=advice_provider,
        embedding_model=EMBEDDING_MODEL,
        vector_candidate_top_k=vector_candidate_top_k,
        bm25_candidate_top_k=bm25_candidate_top_k,
        context_top_k=context_top_k,
        rrf_k=rrf_k,
        min_vector_score=min_vector_score,
    )


@pytest.mark.asyncio
async def test_get_repair_advice_orchestrates_hybrid_rag_pipeline() -> None:
    message = "Что сделать перед укладкой плитки в ванной?"
    waterproofing = make_search_result("tile-waterproofing", score=0.91)
    adhesive = make_search_result("tile-adhesive", score=0.82)
    grout = make_search_result("tile-grout", score=4.2)
    embedding_provider = FakeEmbeddingProvider()
    search_store = FakeSearchStore(
        vector_results=[waterproofing, adhesive],
        bm25_results=[adhesive, grout],
    )
    advice_result = make_advice_result()
    advice_provider = FakeGroundedAdviceProvider(advice_result)
    service = make_service(
        embedding_provider,
        search_store,
        advice_provider,
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        context_top_k=CONTEXT_TOP_K,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
    )

    result = await service.get_repair_advice(message, category=None)

    assert result.status is RagAnswerStatus.ANSWERED
    assert result.advice == advice_result.advice
    assert result.model == advice_result.model
    assert result.usage == advice_result.usage
    assert [source.chunk.id for source in result.sources] == [
        "tile-adhesive",
        "tile-waterproofing",
        "tile-grout",
    ]
    assert result.sources[0].score == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert embedding_provider.calls == [[message]]
    assert search_store.vector_calls == [
        (QUERY_EMBEDDING, EMBEDDING_MODEL, VECTOR_CANDIDATE_TOP_K, None)
    ]
    assert search_store.bm25_calls == [(message, BM25_CANDIDATE_TOP_K, None)]
    assert len(advice_provider.calls) == 1
    called_message, serialized_context = advice_provider.calls[0]
    assert called_message == message
    assert [item["id"] for item in json.loads(serialized_context)] == [
        "tile-adhesive",
        "tile-waterproofing",
        "tile-grout",
    ]


@pytest.mark.asyncio
async def test_passes_category_and_candidate_limits_to_both_searches() -> None:
    message = "Как подготовить ванную?"
    search_store = FakeSearchStore()
    service = make_service(
        FakeEmbeddingProvider(),
        search_store,
        FakeGroundedAdviceProvider(make_advice_result()),
        vector_candidate_top_k=7,
        bm25_candidate_top_k=11,
        context_top_k=3,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
    )

    await service.get_repair_advice(message, category="tile")

    assert search_store.vector_calls == [(QUERY_EMBEDDING, EMBEDDING_MODEL, 7, "tile")]
    assert search_store.bm25_calls == [(message, 11, "tile")]


@pytest.mark.asyncio
async def test_filters_vector_results_before_rrf() -> None:
    relevant = make_search_result("relevant", score=0.75)
    boundary = make_search_result("boundary", score=MIN_VECTOR_SCORE)
    weak = make_search_result("weak", score=MIN_VECTOR_SCORE - 0.01)
    advice_provider = FakeGroundedAdviceProvider(make_advice_result())
    service = make_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(
            vector_results=[relevant, boundary, weak],
            bm25_results=[],
        ),
        advice_provider,
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        context_top_k=CONTEXT_TOP_K,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
    )

    result = await service.get_repair_advice("Вопрос", category=None)

    assert [source.chunk.id for source in result.sources] == ["relevant", "boundary"]
    _, serialized_context = advice_provider.calls[0]
    assert [item["id"] for item in json.loads(serialized_context)] == [
        "relevant",
        "boundary",
    ]


@pytest.mark.asyncio
async def test_keeps_bm25_match_when_vector_score_is_below_threshold() -> None:
    exact_match = make_search_result("tile-adhesive-c2te-s1", score=0.2)
    service = make_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(
            vector_results=[exact_match],
            bm25_results=[exact_match],
        ),
        FakeGroundedAdviceProvider(make_advice_result()),
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        context_top_k=CONTEXT_TOP_K,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
    )

    result = await service.get_repair_advice("C2TE S1", category=None)

    assert [source.chunk.id for source in result.sources] == ["tile-adhesive-c2te-s1"]
    assert result.sources[0].score == pytest.approx(1 / (RRF_K + 1))


@pytest.mark.asyncio
async def test_context_top_k_limits_fused_results() -> None:
    vector_results = [
        make_search_result(f"chunk-{index}", score=0.9 - index / 100)
        for index in range(5)
    ]
    service = make_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(vector_results=vector_results, bm25_results=[]),
        FakeGroundedAdviceProvider(make_advice_result()),
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        context_top_k=2,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
    )

    result = await service.get_repair_advice("Вопрос", category=None)

    assert [source.chunk.id for source in result.sources] == ["chunk-0", "chunk-1"]


@pytest.mark.asyncio
async def test_empty_hybrid_retrieval_returns_abstention_without_llm() -> None:
    search_store = FakeSearchStore()
    advice_provider = FakeGroundedAdviceProvider(make_advice_result())
    service = make_service(
        FakeEmbeddingProvider(),
        search_store,
        advice_provider,
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        context_top_k=CONTEXT_TOP_K,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
    )

    result = await service.get_repair_advice("Неизвестный вопрос", category=None)

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


@pytest.mark.asyncio
async def test_abstains_when_vectors_are_weak_and_bm25_returns_nothing() -> None:
    weak = make_search_result("weak", score=MIN_VECTOR_SCORE - 0.01)
    advice_provider = FakeGroundedAdviceProvider(make_advice_result())
    service = make_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(vector_results=[weak], bm25_results=[]),
        advice_provider,
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        context_top_k=CONTEXT_TOP_K,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
    )

    result = await service.get_repair_advice("Wi-Fi роутер", category=None)

    assert result.status is RagAnswerStatus.INSUFFICIENT_CONTEXT
    assert result.model is None
    assert result.usage is None
    assert result.sources == []
    assert advice_provider.calls == []


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        ("vector_candidate_top_k", "vector_candidate_top_k must be positive"),
        ("bm25_candidate_top_k", "bm25_candidate_top_k must be positive"),
        ("context_top_k", "context_top_k must be positive"),
        ("rrf_k", "rrf_k must be positive"),
    ],
)
@pytest.mark.parametrize("value", [-1, 0])
def test_rejects_non_positive_integer_configuration(
    parameter: str,
    message: str,
    value: int,
) -> None:
    parameters = {
        "vector_candidate_top_k": VECTOR_CANDIDATE_TOP_K,
        "bm25_candidate_top_k": BM25_CANDIDATE_TOP_K,
        "context_top_k": CONTEXT_TOP_K,
        "rrf_k": RRF_K,
    }
    parameters[parameter] = value

    with pytest.raises(ValueError, match=message):
        RagService(
            embedding_provider=FakeEmbeddingProvider(),
            search_store=FakeSearchStore(),
            advice_provider=FakeGroundedAdviceProvider(make_advice_result()),
            embedding_model=EMBEDDING_MODEL,
            vector_candidate_top_k=parameters["vector_candidate_top_k"],
            bm25_candidate_top_k=parameters["bm25_candidate_top_k"],
            context_top_k=parameters["context_top_k"],
            rrf_k=parameters["rrf_k"],
            min_vector_score=MIN_VECTOR_SCORE,
        )


@pytest.mark.parametrize("min_vector_score", [-1.01, 1.01])
def test_rejects_min_vector_score_outside_cosine_similarity_range(
    min_vector_score: float,
) -> None:
    with pytest.raises(ValueError, match="min_vector_score must be between -1 and 1"):
        RagService(
            embedding_provider=FakeEmbeddingProvider(),
            search_store=FakeSearchStore(),
            advice_provider=FakeGroundedAdviceProvider(make_advice_result()),
            embedding_model=EMBEDDING_MODEL,
            vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
            bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
            context_top_k=CONTEXT_TOP_K,
            rrf_k=RRF_K,
            min_vector_score=min_vector_score,
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
    service = make_service(
        FakeEmbeddingProvider(embeddings),
        search_store,
        advice_provider,
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        context_top_k=CONTEXT_TOP_K,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
    )

    with pytest.raises(ValueError):
        await service.get_repair_advice("Вопрос", category=None)

    assert search_store.vector_calls == []
    assert search_store.bm25_calls == []
    assert advice_provider.calls == []
