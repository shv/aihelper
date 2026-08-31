from collections.abc import Awaitable, Callable

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.rag.models import SearchResult

AsyncSearch = Callable[[str], Awaitable[list[SearchResult]]]


class SearchResultLangChainRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retrieve_results: AsyncSearch

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError(
            "SearchResultLangChainRetriever supports only asynchronous retrieval."
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = await self.retrieve_results(query)
        return [
            Document(
                id=result.chunk.id,
                page_content=result.chunk.text,
                metadata={
                    **result.chunk.metadata,
                    "title": result.chunk.title,
                    "score": result.score,
                },
            )
            for result in results
        ]
