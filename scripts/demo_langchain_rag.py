import asyncio
import json
import os

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.llm.prompts import GROUNDED_REPAIR_ASSISTANT_INSTRUCTIONS
from app.rag.citation import validate_citations
from app.rag.langchain_retriever import SearchResultLangChainRetriever
from app.rag.models import SearchResult
from app.schemas import GroundedRepairAdvice
from main import (
    MODEL,
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


def build_grounded_messages(
    inputs: dict[str, object],
) -> list[BaseMessage]:
    question = inputs["question"]
    raw_documents = inputs["documents"]

    if not isinstance(question, str):
        raise TypeError("question must be a string")

    if not isinstance(raw_documents, list):
        raise TypeError("documents must be a list")

    documents: list[Document] = []

    for document in raw_documents:
        if not isinstance(document, Document):
            raise TypeError("documents must contain Document instances")

        if document.id is None:
            raise ValueError("document id must not be None")

        documents.append(document)

    context = [
        {
            "id": document.id,
            "title": document.metadata["title"],
            "text": document.page_content,
        }
        for document in documents
    ]

    input_payload = json.dumps(
        {
            "question": question,
            "context": context,
        },
        ensure_ascii=False,
    )

    return [
        SystemMessage(
            content=GROUNDED_REPAIR_ASSISTANT_INSTRUCTIONS,
        ),
        HumanMessage(
            content=input_payload,
        ),
    ]


async def main() -> None:
    openai_client = get_openai_client()

    service = get_rag_service(
        embedding_provider=get_embedding_provider(openai_client),
        search_store=get_search_store(),
        reranker=get_reranker(openai_client),
        token_counter=get_token_counter(),
        advice_provider=get_grounded_repair_advice_provider(openai_client),
    )

    captured_results: list[SearchResult] | None = None

    async def retrieve_results(query: str) -> list[SearchResult]:
        nonlocal captured_results

        captured_results = await service.retrieve(
            query,
            category=CATEGORY,
        )
        return captured_results

    retriever = SearchResultLangChainRetriever(
        retrieve_results=retrieve_results,
    )

    model = ChatOpenAI(
        model=MODEL,
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        timeout=20.0,
        max_retries=2,
        reasoning_effort="low",
        use_responses_api=True,
    )

    structured_model = model.with_structured_output(
        GroundedRepairAdvice,
        method="json_schema",
        include_raw=False,
        strict=True,
    )

    rag_chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            documents=retriever,
        )
        | RunnableLambda(
            build_grounded_messages,
            name="build_grounded_messages",
        )
        | structured_model
    )

    try:
        result = await rag_chain.ainvoke(QUERY)
    finally:
        await openai_client.close()

    if not isinstance(result, GroundedRepairAdvice):
        raise TypeError("LangChain returned unexpected structured output")

    if captured_results is None:
        raise RuntimeError("Retriever was not called")

    validate_citations(
        result.citations,
        captured_results,
    )

    print(result.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
