import asyncio
from dataclasses import dataclass
from typing import Protocol

from app.llm.base import GroundedRepairAdviceProvider
from app.rag.citation import validate_citations
from app.rag.context import build_rag_context
from app.rag.embeddings import EmbeddingProvider
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.models import SearchResult
from app.rag.reranker import Reranker
from app.schemas import RagAnswerStatus, RagCitation, RepairAdvice, TokenUsage


class SearchStore(Protocol):
    async def search(
        self,
        query_embedding: list[float],
        *,
        embedding_model: str,
        top_k: int,
        category: str | None = None,
    ) -> list[SearchResult]: ...

    async def search_bm25(
        self, query: str, *, top_k: int, category: str | None = None
    ) -> list[SearchResult]: ...


@dataclass(frozen=True, slots=True)
class GroundedRepairAdviceResult:
    status: RagAnswerStatus
    advice: RepairAdvice
    citations: list[RagCitation]
    model: str | None
    usage: TokenUsage | None
    sources: list[SearchResult]


class RagService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        search_store: SearchStore,
        reranker: Reranker,
        advice_provider: GroundedRepairAdviceProvider,
        *,
        embedding_model: str,
        vector_candidate_top_k: int,
        bm25_candidate_top_k: int,
        rerank_candidate_top_k: int,
        context_top_k: int,
        rrf_k: int,
        min_vector_score: float,
        min_rerank_score: float,
    ) -> None:
        if vector_candidate_top_k <= 0:
            raise ValueError("vector_candidate_top_k must be positive")

        if bm25_candidate_top_k <= 0:
            raise ValueError("bm25_candidate_top_k must be positive")

        if rerank_candidate_top_k <= 0:
            raise ValueError("rerank_candidate_top_k must be positive")

        if context_top_k <= 0:
            raise ValueError("context_top_k must be positive")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")

        if not -1.0 <= min_vector_score <= 1.0:
            raise ValueError("min_vector_score must be between -1 and 1")

        if not 0.0 <= min_rerank_score <= 3.0:
            raise ValueError("min_rerank_score must be between 0 and 3")

        if context_top_k > rerank_candidate_top_k:
            raise ValueError("context_top_k must not exceed rerank_candidate_top_k")

        self._embedding_provider = embedding_provider
        self._search_store = search_store
        self._reranker = reranker
        self._advice_provider = advice_provider
        self._embedding_model = embedding_model
        self._vector_candidate_top_k = vector_candidate_top_k
        self._bm25_candidate_top_k = bm25_candidate_top_k
        self._rerank_candidate_top_k = rerank_candidate_top_k
        self._context_top_k = context_top_k
        self._rrf_k = rrf_k
        self._min_vector_score = min_vector_score
        self._min_rerank_score = min_rerank_score

    async def get_repair_advice(
        self, message: str, *, category: str | None = None
    ) -> GroundedRepairAdviceResult:
        [query_embedding] = await self._embedding_provider.embed([message])

        vector_results, bm25_result = await asyncio.gather(
            self._search_store.search(
                query_embedding,
                embedding_model=self._embedding_model,
                top_k=self._vector_candidate_top_k,
                category=category,
            ),
            self._search_store.search_bm25(
                message,
                top_k=self._bm25_candidate_top_k,
                category=category,
            ),
        )

        relevant_vector_results = [
            result
            for result in vector_results
            if result.score >= self._min_vector_score
        ]

        fused_candidates = reciprocal_rank_fusion(
            [relevant_vector_results, bm25_result],
            rrf_k=self._rrf_k,
            top_k=self._rerank_candidate_top_k,
        )

        reranked_candidates = await self._reranker.rerank(message, fused_candidates)

        relevant_results = [
            candidate
            for candidate in reranked_candidates
            if candidate.score >= self._min_rerank_score
        ][: self._context_top_k]

        if not relevant_results:
            return GroundedRepairAdviceResult(
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

        context = build_rag_context(relevant_results)

        advice_result = await self._advice_provider.get_grounded_repair_advice(
            message=message, context=context
        )

        validate_citations(advice_result.citations, relevant_results)

        return GroundedRepairAdviceResult(
            status=RagAnswerStatus.ANSWERED,
            advice=advice_result.advice,
            citations=advice_result.citations,
            model=advice_result.model,
            usage=advice_result.usage,
            sources=relevant_results,
        )
