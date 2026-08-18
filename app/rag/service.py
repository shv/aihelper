from dataclasses import dataclass
from typing import Protocol

from app.llm.base import GroundedRepairAdviceProvider, RepairAdviceResult
from app.rag.context import build_rag_context
from app.rag.embeddings import EmbeddingProvider
from app.rag.models import SearchResult


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
    advice_result: RepairAdviceResult
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
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self._embedding_provider = embedding_provider
        self._search_store = search_store
        self._advice_provider = advice_provider
        self._embedding_model = embedding_model
        self._top_k = top_k

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

        context = build_rag_context(search_result)

        advice_result = await self._advice_provider.get_grounded_repair_advice(
            message=message, context=context
        )

        return GroundedRepairAdviceResult(
            advice_result=advice_result, sources=search_result
        )
