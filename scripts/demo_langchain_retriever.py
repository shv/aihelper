import asyncio
import json

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.rag.langchain_retriever import SearchResultLangChainRetriever
from app.rag.models import SearchResult
from main import (
    get_embedding_provider,
    get_grounded_repair_advice_provider,
    get_openai_client,
    get_rag_service,
    get_reranker,
    get_search_store,
    get_token_counter,
)

QUERY = "Что означает маркировка плиточного клея C2TE S1?"
CATEGORY = "tile"


def documents_to_payload(documents: list[Document]) -> list[dict[str, object]]:
    return [
        {
            "id": document.id,
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        for document in documents
    ]


async def main() -> None:
    client = get_openai_client()

    service = get_rag_service(
        embedding_provider=get_embedding_provider(client),
        search_store=get_search_store(),
        reranker=get_reranker(client),
        token_counter=get_token_counter(),
        advice_provider=get_grounded_repair_advice_provider(client),
    )

    captured_results: list[SearchResult] | None = None

    async def retrieve_results(query: str) -> list[SearchResult]:
        nonlocal captured_results

        captured_results = await service.retrieve(query, category=CATEGORY)
        return captured_results

    retriever = SearchResultLangChainRetriever(retrieve_results=retrieve_results)

    try:
        retrieval_chain = retriever | RunnableLambda(
            documents_to_payload, name="documents_to_payload"
        )
        langchain_payload = await retrieval_chain.ainvoke(QUERY)
    finally:
        await client.close()

    if captured_results is None:
        raise RuntimeError("LangChain retriever did not call retrieve_results")

    payload = {
        "query": QUERY,
        "category": CATEGORY,
        "native": [
            {
                "id": result.chunk.id,
                "title": result.chunk.title,
                "text": result.chunk.text,
                "metadata": result.chunk.metadata,
                "score": result.score,
            }
            for result in captured_results
        ],
        "langchain": langchain_payload,
    }

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
