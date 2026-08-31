from collections.abc import Awaitable, Callable

import pytest

from app.rag.langchain_retriever import SearchResultLangChainRetriever
from app.rag.models import DocumentChunk, SearchResult


def make_retrieve_results(
    results: list[SearchResult],
) -> tuple[Callable[[str], Awaitable[list[SearchResult]]], list[str]]:
    received_queries: list[str] = []

    async def retrieve_results(query: str) -> list[SearchResult]:
        received_queries.append(query)
        return results

    return retrieve_results, received_queries


@pytest.mark.asyncio
async def test_ainvoke_passes_query_and_converts_search_results_to_documents() -> None:
    retrieve_results, received_queries = make_retrieve_results(
        [
            SearchResult(
                chunk=DocumentChunk(
                    id="tile-waterproofing",
                    title="Подготовка мокрой зоны",
                    text="Основание очищают, грунтуют и выполняют гидроизоляцию.",
                    metadata={"category": "tile"},
                ),
                score=0.75,
            )
        ]
    )
    retriever = SearchResultLangChainRetriever(
        retrieve_results=retrieve_results,
    )

    documents = await retriever.ainvoke("Как подготовить стены ванной?")

    assert received_queries == ["Как подготовить стены ванной?"]
    assert len(documents) == 1
    assert documents[0].id == "tile-waterproofing"
    assert (
        documents[0].page_content
        == "Основание очищают, грунтуют и выполняют гидроизоляцию."
    )
    assert documents[0].metadata == {
        "category": "tile",
        "title": "Подготовка мокрой зоны",
        "score": 0.75,
    }


@pytest.mark.asyncio
async def test_ainvoke_preserves_result_order() -> None:
    retrieve_results, _ = make_retrieve_results(
        [
            SearchResult(
                chunk=DocumentChunk(
                    id="first",
                    title="Первый",
                    text="Первый результат",
                    metadata={},
                ),
                score=0.9,
            ),
            SearchResult(
                chunk=DocumentChunk(
                    id="second",
                    title="Второй",
                    text="Второй результат",
                    metadata={},
                ),
                score=0.8,
            ),
        ]
    )
    retriever = SearchResultLangChainRetriever(
        retrieve_results=retrieve_results,
    )

    documents = await retriever.ainvoke("запрос")

    assert [document.id for document in documents] == ["first", "second"]


@pytest.mark.asyncio
async def test_ainvoke_overrides_reserved_metadata_with_search_result_values() -> None:
    retrieve_results, _ = make_retrieve_results(
        [
            SearchResult(
                chunk=DocumentChunk(
                    id="chunk-id",
                    title="Настоящий заголовок",
                    text="Текст",
                    metadata={
                        "title": "Устаревший заголовок",
                        "score": "устаревшая оценка",
                    },
                ),
                score=0.6,
            )
        ]
    )
    retriever = SearchResultLangChainRetriever(
        retrieve_results=retrieve_results,
    )

    documents = await retriever.ainvoke("запрос")

    assert documents[0].metadata == {
        "title": "Настоящий заголовок",
        "score": 0.6,
    }


def test_invoke_rejects_synchronous_retrieval() -> None:
    retrieve_results, _ = make_retrieve_results([])
    retriever = SearchResultLangChainRetriever(
        retrieve_results=retrieve_results,
    )

    with pytest.raises(
        NotImplementedError,
        match="supports only asynchronous retrieval",
    ):
        retriever.invoke("запрос")
