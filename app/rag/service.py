from dataclasses import dataclass
from typing import Protocol

from app.llm.base import GroundedRepairAdviceProvider
from app.rag.context import build_rag_context
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import SearchResult
from app.schemas import RagAnswerStatus, RepairAdvice, TokenUsage


class SearchStore(Protocol):
    async def search(
        self,
        query_embedding: list[float],
        *,
        embedding_model: str,
        top_k: int,
        category: str | None = None,
    ) -> list[SearchResult]: ...


@dataclass(frozen=True, slots=True)
class GroundedRepairAdviceResult:
    status: RagAnswerStatus
    advice: RepairAdvice
    model: str | None
    usage: TokenUsage | None
    sources: list[SearchResult]


class RagService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        search_store: SearchStore,
        advice_provider: GroundedRepairAdviceProvider,
        *,
        embedding_model: str,
        top_k: int,
        min_score: float,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1")

        self._embedding_provider = embedding_provider
        self._search_store = search_store
        self._advice_provider = advice_provider
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._min_score = min_score

    async def get_repair_advice(
        self, message: str, *, category: str | None = None
    ) -> GroundedRepairAdviceResult:
        [query_embedding] = await self._embedding_provider.embed([message])

        search_result = await self._search_store.search(
            query_embedding,
            embedding_model=self._embedding_model,
            top_k=self._top_k,
            category=category,
        )

        relevant_results = [
            result for result in search_result if result.score >= self._min_score
        ]

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
                model=None,
                usage=None,
                sources=[],
            )

        context = build_rag_context(relevant_results)

        advice_result = await self._advice_provider.get_grounded_repair_advice(
            message=message, context=context
        )

        return GroundedRepairAdviceResult(
            status=RagAnswerStatus.ANSWERED,
            advice=advice_result.advice,
            model=advice_result.model,
            usage=advice_result.usage,
            sources=relevant_results,
        )
