import json
from dataclasses import dataclass, field

import pytest

from app.llm.base import GroundedRepairAdviceProviderResult
from app.llm.exceptions import LLMInvalidResponseError
from app.rag.context import serialize_context_payload
from app.rag.models import DocumentChunk, SearchResult
from app.rag.service import GroundedRepairAdviceResult, RagService
from app.schemas import RagAnswerStatus, RagCitation, RepairAdvice, TokenUsage

EMBEDDING_MODEL = "test-embedding-model"
QUERY_EMBEDDING = [0.1, 0.2, 0.3]
VECTOR_CANDIDATE_TOP_K = 10
BM25_CANDIDATE_TOP_K = 8
RERANK_CANDIDATE_TOP_K = 6
CONTEXT_TOP_K = 3
RRF_K = 60
MIN_VECTOR_SCORE = 0.45
MIN_RERANK_SCORE = 2.0
CONTEXT_MAX_TOKENS = 10_000


class FakeEmbeddingProvider:
    def __init__(self, embeddings: list[list[float]] | None = None) -> None:
        self._embeddings = embeddings if embeddings is not None else [QUERY_EMBEDDING]
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self._embeddings


@dataclass
class CharacterTokenCounter:
    calls: list[str] = field(default_factory=list)

    def count_tokens(self, text: str) -> int:
        self.calls.append(text)
        return len(text)


@dataclass
class FakeSearchStore:
    vector_results: list[SearchResult]
    bm25_results: list[SearchResult]
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
class FakeReranker:
    results: list[SearchResult]
    calls: list[tuple[str, list[SearchResult]]] = field(default_factory=list)

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        self.calls.append((query, candidates))
        return self.results


@dataclass
class FakeGroundedAdviceProvider:
    result: GroundedRepairAdviceProviderResult
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def get_grounded_repair_advice(
        self,
        message: str,
        context: str,
    ) -> GroundedRepairAdviceProviderResult:
        self.calls.append((message, context))
        return self.result


def make_advice_result(
    source_id: str,
    quote: str,
) -> GroundedRepairAdviceProviderResult:
    return GroundedRepairAdviceProviderResult(
        advice=RepairAdvice(
            summary="Использовать гидроизоляцию",
            clarifying_questions=[],
            recommendations=["Подготовить основание"],
            risks=[],
            requires_professional=False,
        ),
        model="fake-model",
        usage=TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        citations=[RagCitation(source_id=source_id, quote=quote)],
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
    reranker: FakeReranker,
    token_counter: CharacterTokenCounter,
    advice_provider: FakeGroundedAdviceProvider,
    *,
    vector_candidate_top_k: int,
    bm25_candidate_top_k: int,
    rerank_candidate_top_k: int,
    context_top_k: int,
    context_max_tokens: int,
    rrf_k: int,
    min_vector_score: float,
    min_rerank_score: float,
) -> RagService:
    return RagService(
        embedding_provider=embedding_provider,
        search_store=search_store,
        reranker=reranker,
        token_counter=token_counter,
        advice_provider=advice_provider,
        embedding_model=EMBEDDING_MODEL,
        vector_candidate_top_k=vector_candidate_top_k,
        bm25_candidate_top_k=bm25_candidate_top_k,
        rerank_candidate_top_k=rerank_candidate_top_k,
        context_top_k=context_top_k,
        context_max_tokens=context_max_tokens,
        rrf_k=rrf_k,
        min_vector_score=min_vector_score,
        min_rerank_score=min_rerank_score,
    )


def make_default_service(
    embedding_provider: FakeEmbeddingProvider,
    search_store: FakeSearchStore,
    reranker: FakeReranker,
    advice_provider: FakeGroundedAdviceProvider,
) -> RagService:
    return make_service(
        embedding_provider,
        search_store,
        reranker,
        CharacterTokenCounter(),
        advice_provider,
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        rerank_candidate_top_k=RERANK_CANDIDATE_TOP_K,
        context_top_k=CONTEXT_TOP_K,
        context_max_tokens=CONTEXT_MAX_TOKENS,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
        min_rerank_score=MIN_RERANK_SCORE,
    )


@pytest.mark.asyncio
async def test_retrieve_runs_hybrid_pipeline_without_generating_advice() -> None:
    message = "Что означает C2TE S1?"
    vector_match = make_search_result("vector-match", score=0.75)
    weak_vector_match = make_search_result(
        "weak-vector-match",
        score=MIN_VECTOR_SCORE - 0.01,
    )
    bm25_match = make_search_result("bm25-match", score=5.0)
    reranked_results = [
        make_search_result("bm25-match", score=3.0),
        make_search_result("vector-match", score=2.0),
        make_search_result("irrelevant", score=1.0),
    ]
    embedding_provider = FakeEmbeddingProvider()
    search_store = FakeSearchStore(
        vector_results=[vector_match, weak_vector_match],
        bm25_results=[bm25_match],
    )
    reranker = FakeReranker(results=reranked_results)
    token_counter = CharacterTokenCounter()
    advice_provider = FakeGroundedAdviceProvider(
        make_advice_result("bm25-match", "Text bm25-match")
    )
    service = make_service(
        embedding_provider,
        search_store,
        reranker,
        token_counter,
        advice_provider,
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        rerank_candidate_top_k=RERANK_CANDIDATE_TOP_K,
        context_top_k=CONTEXT_TOP_K,
        context_max_tokens=CONTEXT_MAX_TOKENS,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
        min_rerank_score=MIN_RERANK_SCORE,
    )

    results = await service.retrieve(message, category="tile")

    assert results == reranked_results[:2]
    assert embedding_provider.calls == [[message]]
    assert search_store.vector_calls == [
        (QUERY_EMBEDDING, EMBEDDING_MODEL, VECTOR_CANDIDATE_TOP_K, "tile")
    ]
    assert search_store.bm25_calls == [(message, BM25_CANDIDATE_TOP_K, "tile")]
    assert len(reranker.calls) == 1
    rerank_query, fused_candidates = reranker.calls[0]
    assert rerank_query == message
    assert [candidate.chunk.id for candidate in fused_candidates] == [
        "bm25-match",
        "vector-match",
    ]
    assert token_counter.calls == []
    assert advice_provider.calls == []


@pytest.mark.asyncio
async def test_get_repair_advice_orchestrates_reranked_hybrid_pipeline() -> None:
    message = "Что означает C2TE S1?"
    explanation = make_search_result("c2te-s1", score=0.60)
    label = make_search_result("tile-adhesive-c2te-s1", score=0.55)
    c1 = make_search_result("tile-adhesive-c1", score=4.0)
    reranked = [
        make_search_result("c2te-s1", score=3.0),
        make_search_result("tile-adhesive-c2te-s1", score=2.0),
        make_search_result("tile-adhesive-c1", score=0.0),
    ]
    embedding_provider = FakeEmbeddingProvider()
    search_store = FakeSearchStore(
        vector_results=[explanation, label],
        bm25_results=[label, c1, explanation],
    )
    reranker = FakeReranker(results=reranked)
    advice_result = make_advice_result("c2te-s1", "Text c2te-s1")
    advice_provider = FakeGroundedAdviceProvider(advice_result)
    service = make_default_service(
        embedding_provider,
        search_store,
        reranker,
        advice_provider,
    )

    result = await service.get_repair_advice(message, category="tile")

    assert result.status is RagAnswerStatus.ANSWERED
    assert result.advice == advice_result.advice
    assert result.citations == advice_result.citations
    assert result.model == advice_result.model
    assert result.usage == advice_result.usage
    assert [(source.chunk.id, source.score) for source in result.sources] == [
        ("c2te-s1", 3.0),
        ("tile-adhesive-c2te-s1", 2.0),
    ]
    assert embedding_provider.calls == [[message]]
    assert search_store.vector_calls == [
        (QUERY_EMBEDDING, EMBEDDING_MODEL, VECTOR_CANDIDATE_TOP_K, "tile")
    ]
    assert search_store.bm25_calls == [(message, BM25_CANDIDATE_TOP_K, "tile")]
    assert len(reranker.calls) == 1
    rerank_query, fused_candidates = reranker.calls[0]
    assert rerank_query == message
    assert [candidate.chunk.id for candidate in fused_candidates] == [
        "tile-adhesive-c2te-s1",
        "c2te-s1",
        "tile-adhesive-c1",
    ]
    assert len(advice_provider.calls) == 1
    called_message, serialized_context = advice_provider.calls[0]
    assert called_message == message
    assert [item["id"] for item in json.loads(serialized_context)] == [
        "c2te-s1",
        "tile-adhesive-c2te-s1",
    ]


@pytest.mark.asyncio
async def test_filters_vector_results_before_rrf_and_reranking() -> None:
    relevant = make_search_result("relevant", score=0.75)
    boundary = make_search_result("boundary", score=MIN_VECTOR_SCORE)
    weak = make_search_result("weak", score=MIN_VECTOR_SCORE - 0.01)
    reranker = FakeReranker(
        results=[
            make_search_result("relevant", score=3.0),
            make_search_result("boundary", score=2.0),
        ]
    )
    service = make_default_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(
            vector_results=[relevant, boundary, weak],
            bm25_results=[],
        ),
        reranker,
        FakeGroundedAdviceProvider(make_advice_result("relevant", "Text relevant")),
    )

    result = await service.get_repair_advice("Вопрос", category=None)

    assert [candidate.chunk.id for candidate in reranker.calls[0][1]] == [
        "relevant",
        "boundary",
    ]
    assert [source.chunk.id for source in result.sources] == ["relevant", "boundary"]


@pytest.mark.asyncio
async def test_bm25_match_survives_weak_vector_score_and_reaches_reranker() -> None:
    exact_match = make_search_result("tile-adhesive-c2te-s1", score=0.2)
    reranker = FakeReranker(
        results=[make_search_result("tile-adhesive-c2te-s1", score=3.0)]
    )
    service = make_default_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(
            vector_results=[exact_match],
            bm25_results=[exact_match],
        ),
        reranker,
        FakeGroundedAdviceProvider(
            make_advice_result(
                "tile-adhesive-c2te-s1",
                "Text tile-adhesive-c2te-s1",
            )
        ),
    )

    result = await service.get_repair_advice("C2TE S1", category=None)

    assert [candidate.chunk.id for candidate in reranker.calls[0][1]] == [
        "tile-adhesive-c2te-s1"
    ]
    assert [(source.chunk.id, source.score) for source in result.sources] == [
        ("tile-adhesive-c2te-s1", 3.0)
    ]


@pytest.mark.asyncio
async def test_applies_rerank_threshold_before_context_top_k() -> None:
    vector_results = [
        make_search_result(f"chunk-{index}", score=0.9 - index / 100)
        for index in range(5)
    ]
    reranker = FakeReranker(
        results=[
            make_search_result("chunk-4", score=3.0),
            make_search_result("chunk-3", score=2.5),
            make_search_result("chunk-2", score=2.0),
            make_search_result("chunk-1", score=1.0),
            make_search_result("chunk-0", score=0.0),
        ]
    )
    service = make_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(vector_results=vector_results, bm25_results=[]),
        reranker,
        CharacterTokenCounter(),
        FakeGroundedAdviceProvider(make_advice_result("chunk-4", "Text chunk-4")),
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        rerank_candidate_top_k=RERANK_CANDIDATE_TOP_K,
        context_top_k=2,
        context_max_tokens=CONTEXT_MAX_TOKENS,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
        min_rerank_score=MIN_RERANK_SCORE,
    )

    result = await service.get_repair_advice("Вопрос", category=None)

    assert len(reranker.calls[0][1]) == 5
    assert [source.chunk.id for source in result.sources] == ["chunk-4", "chunk-3"]


@pytest.mark.asyncio
async def test_empty_retrieval_passes_empty_candidates_and_abstains() -> None:
    reranker = FakeReranker(results=[])
    advice_provider = FakeGroundedAdviceProvider(
        make_advice_result("placeholder", "Text placeholder")
    )
    service = make_default_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(vector_results=[], bm25_results=[]),
        reranker,
        advice_provider,
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
        citations=[],
        model=None,
        usage=None,
        sources=[],
    )
    assert reranker.calls == [("Неизвестный вопрос", [])]
    assert advice_provider.calls == []


@pytest.mark.asyncio
async def test_abstains_when_all_reranked_candidates_are_below_threshold() -> None:
    candidate = make_search_result("topical-only", score=0.8)
    reranker = FakeReranker(
        results=[make_search_result("topical-only", score=MIN_RERANK_SCORE - 1)]
    )
    advice_provider = FakeGroundedAdviceProvider(
        make_advice_result("placeholder", "Text placeholder")
    )
    service = make_default_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(vector_results=[candidate], bm25_results=[]),
        reranker,
        advice_provider,
    )

    result = await service.get_repair_advice("Вопрос", category=None)

    assert result.status is RagAnswerStatus.INSUFFICIENT_CONTEXT
    assert result.model is None
    assert result.usage is None
    assert result.sources == []
    assert advice_provider.calls == []


@pytest.mark.asyncio
async def test_rejects_provider_citation_not_present_in_final_context() -> None:
    candidate = make_search_result("c2te-s1", score=0.8)
    advice_provider = FakeGroundedAdviceProvider(
        make_advice_result("c2te-s1", "Выдуманная цитата")
    )
    service = make_default_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(vector_results=[candidate], bm25_results=[]),
        FakeReranker(results=[make_search_result("c2te-s1", score=3.0)]),
        advice_provider,
    )

    with pytest.raises(
        LLMInvalidResponseError,
        match="Citation is not an exact quote from source: c2te-s1",
    ):
        await service.get_repair_advice("Что означает C2TE S1?", category=None)

    assert len(advice_provider.calls) == 1


@pytest.mark.asyncio
async def test_context_budget_skips_oversized_source_before_provider() -> None:
    oversized = SearchResult(
        chunk=DocumentChunk(
            id="oversized",
            title="Oversized",
            text="x" * 500,
            metadata={"category": "tile"},
        ),
        score=3.0,
    )
    small = make_search_result("small", score=3.0)
    small_context = serialize_context_payload([small])
    token_counter = CharacterTokenCounter()
    advice_provider = FakeGroundedAdviceProvider(
        make_advice_result("small", "Text small")
    )
    service = make_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(vector_results=[oversized, small], bm25_results=[]),
        FakeReranker(results=[oversized, small]),
        token_counter,
        advice_provider,
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        rerank_candidate_top_k=RERANK_CANDIDATE_TOP_K,
        context_top_k=CONTEXT_TOP_K,
        context_max_tokens=len(small_context),
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
        min_rerank_score=MIN_RERANK_SCORE,
    )

    result = await service.get_repair_advice("Вопрос", category=None)

    assert result.sources == [small]
    assert advice_provider.calls == [("Вопрос", small_context)]
    assert any("oversized" in text for text in token_counter.calls)


@pytest.mark.asyncio
async def test_context_budget_abstains_when_no_source_fits() -> None:
    oversized = SearchResult(
        chunk=DocumentChunk(
            id="oversized",
            title="Oversized",
            text="x" * 500,
            metadata={"category": "tile"},
        ),
        score=3.0,
    )
    advice_provider = FakeGroundedAdviceProvider(
        make_advice_result("placeholder", "Text placeholder")
    )
    service = make_service(
        FakeEmbeddingProvider(),
        FakeSearchStore(vector_results=[oversized], bm25_results=[]),
        FakeReranker(results=[oversized]),
        CharacterTokenCounter(),
        advice_provider,
        vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
        bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
        rerank_candidate_top_k=RERANK_CANDIDATE_TOP_K,
        context_top_k=CONTEXT_TOP_K,
        context_max_tokens=2,
        rrf_k=RRF_K,
        min_vector_score=MIN_VECTOR_SCORE,
        min_rerank_score=MIN_RERANK_SCORE,
    )

    result = await service.get_repair_advice("Вопрос", category=None)

    assert result.status is RagAnswerStatus.INSUFFICIENT_CONTEXT
    assert result.sources == []
    assert result.citations == []
    assert advice_provider.calls == []


@pytest.mark.parametrize(
    ("parameter", "message"),
    [
        ("vector_candidate_top_k", "vector_candidate_top_k must be positive"),
        ("bm25_candidate_top_k", "bm25_candidate_top_k must be positive"),
        ("rerank_candidate_top_k", "rerank_candidate_top_k must be positive"),
        ("context_top_k", "context_top_k must be positive"),
        ("context_max_tokens", "context_max_tokens must be positive"),
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
        "rerank_candidate_top_k": RERANK_CANDIDATE_TOP_K,
        "context_top_k": CONTEXT_TOP_K,
        "context_max_tokens": CONTEXT_MAX_TOKENS,
        "rrf_k": RRF_K,
    }
    parameters[parameter] = value

    with pytest.raises(ValueError, match=message):
        RagService(
            embedding_provider=FakeEmbeddingProvider(),
            search_store=FakeSearchStore(vector_results=[], bm25_results=[]),
            reranker=FakeReranker(results=[]),
            token_counter=CharacterTokenCounter(),
            advice_provider=FakeGroundedAdviceProvider(
                make_advice_result("placeholder", "Text placeholder")
            ),
            embedding_model=EMBEDDING_MODEL,
            vector_candidate_top_k=parameters["vector_candidate_top_k"],
            bm25_candidate_top_k=parameters["bm25_candidate_top_k"],
            rerank_candidate_top_k=parameters["rerank_candidate_top_k"],
            context_top_k=parameters["context_top_k"],
            context_max_tokens=parameters["context_max_tokens"],
            rrf_k=parameters["rrf_k"],
            min_vector_score=MIN_VECTOR_SCORE,
            min_rerank_score=MIN_RERANK_SCORE,
        )


def test_rejects_context_top_k_greater_than_rerank_candidate_top_k() -> None:
    with pytest.raises(
        ValueError,
        match="context_top_k must not exceed rerank_candidate_top_k",
    ):
        make_service(
            FakeEmbeddingProvider(),
            FakeSearchStore(vector_results=[], bm25_results=[]),
            FakeReranker(results=[]),
            CharacterTokenCounter(),
            FakeGroundedAdviceProvider(
                make_advice_result("placeholder", "Text placeholder")
            ),
            vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
            bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
            rerank_candidate_top_k=2,
            context_top_k=3,
            context_max_tokens=CONTEXT_MAX_TOKENS,
            rrf_k=RRF_K,
            min_vector_score=MIN_VECTOR_SCORE,
            min_rerank_score=MIN_RERANK_SCORE,
        )


@pytest.mark.parametrize("min_vector_score", [-1.01, 1.01])
def test_rejects_min_vector_score_outside_cosine_range(
    min_vector_score: float,
) -> None:
    with pytest.raises(ValueError, match="min_vector_score must be between -1 and 1"):
        make_service(
            FakeEmbeddingProvider(),
            FakeSearchStore(vector_results=[], bm25_results=[]),
            FakeReranker(results=[]),
            CharacterTokenCounter(),
            FakeGroundedAdviceProvider(
                make_advice_result("placeholder", "Text placeholder")
            ),
            vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
            bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
            rerank_candidate_top_k=RERANK_CANDIDATE_TOP_K,
            context_top_k=CONTEXT_TOP_K,
            context_max_tokens=CONTEXT_MAX_TOKENS,
            rrf_k=RRF_K,
            min_vector_score=min_vector_score,
            min_rerank_score=MIN_RERANK_SCORE,
        )


@pytest.mark.parametrize("min_rerank_score", [-0.01, 3.01])
def test_rejects_min_rerank_score_outside_relevance_range(
    min_rerank_score: float,
) -> None:
    with pytest.raises(ValueError, match="min_rerank_score must be between 0 and 3"):
        make_service(
            FakeEmbeddingProvider(),
            FakeSearchStore(vector_results=[], bm25_results=[]),
            FakeReranker(results=[]),
            CharacterTokenCounter(),
            FakeGroundedAdviceProvider(
                make_advice_result("placeholder", "Text placeholder")
            ),
            vector_candidate_top_k=VECTOR_CANDIDATE_TOP_K,
            bm25_candidate_top_k=BM25_CANDIDATE_TOP_K,
            rerank_candidate_top_k=RERANK_CANDIDATE_TOP_K,
            context_top_k=CONTEXT_TOP_K,
            context_max_tokens=CONTEXT_MAX_TOKENS,
            rrf_k=RRF_K,
            min_vector_score=MIN_VECTOR_SCORE,
            min_rerank_score=min_rerank_score,
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
    search_store = FakeSearchStore(vector_results=[], bm25_results=[])
    reranker = FakeReranker(results=[])
    advice_provider = FakeGroundedAdviceProvider(
        make_advice_result("placeholder", "Text placeholder")
    )
    service = make_default_service(
        FakeEmbeddingProvider(embeddings),
        search_store,
        reranker,
        advice_provider,
    )

    with pytest.raises(ValueError):
        await service.get_repair_advice("Вопрос", category=None)

    assert search_store.vector_calls == []
    assert search_store.bm25_calls == []
    assert reranker.calls == []
    assert advice_provider.calls == []
