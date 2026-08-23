import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from pydantic import ValidationError

from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.rag.models import DocumentChunk, SearchResult
from app.rag.reranker import (
    RERANK_INSTRUCTIONS,
    OpenAIReranker,
    RerankItem,
    RerankResponse,
    build_rerank_input,
)

MODEL = "fake-reranker-model"
REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/responses")


def make_candidate(chunk_id: str, *, score: float) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            title=f"Title {chunk_id}",
            text=f"Text {chunk_id}",
            metadata={"category": "tile"},
        ),
        score=score,
    )


def make_client(parse_mock: AsyncMock) -> AsyncOpenAI:
    client_mock = MagicMock(spec=AsyncOpenAI)
    client_mock.responses.parse = parse_mock
    return cast(AsyncOpenAI, client_mock)


def make_response(items: list[RerankItem]) -> SimpleNamespace:
    return SimpleNamespace(output_parsed=RerankResponse(items=items))


def test_build_rerank_input_separates_query_and_documents() -> None:
    candidate = make_candidate("c2te-s1", score=0.032)

    payload = json.loads(build_rerank_input("Что означает C2TE S1?", [candidate]))

    assert payload == {
        "query": "Что означает C2TE S1?",
        "documents": [
            {
                "chunk_id": "c2te-s1",
                "title": "Title c2te-s1",
                "text": "Text c2te-s1",
            }
        ],
    }
    assert "score" not in payload["documents"][0]


@pytest.mark.parametrize("relevance_score", [0, 1, 2, 3])
def test_rerank_item_accepts_every_documented_score(relevance_score: int) -> None:
    item = RerankItem(chunk_id="chunk", relevance_score=relevance_score)

    assert item.relevance_score == relevance_score


@pytest.mark.parametrize("relevance_score", [-1, 4])
def test_rerank_item_rejects_score_outside_documented_range(
    relevance_score: int,
) -> None:
    with pytest.raises(ValidationError):
        RerankItem(chunk_id="chunk", relevance_score=relevance_score)


@pytest.mark.asyncio
async def test_rerank_orders_by_relevance_and_replaces_rrf_scores() -> None:
    topical = make_candidate("tile-adhesive-c2te-s1", score=0.0328)
    answer = make_candidate("c2te-s1", score=0.0320)
    irrelevant = make_candidate("tile-adhesive-c1", score=0.0320)
    parse_mock = AsyncMock(
        return_value=make_response(
            [
                RerankItem(chunk_id=irrelevant.chunk.id, relevance_score=0),
                RerankItem(chunk_id=answer.chunk.id, relevance_score=3),
                RerankItem(chunk_id=topical.chunk.id, relevance_score=1),
            ]
        )
    )
    reranker = OpenAIReranker(client=make_client(parse_mock), model=MODEL)
    candidates = [topical, answer, irrelevant]

    results = await reranker.rerank("Что означает C2TE S1?", candidates)

    assert [(result.chunk.id, result.score) for result in results] == [
        ("c2te-s1", 3.0),
        ("tile-adhesive-c2te-s1", 1.0),
        ("tile-adhesive-c1", 0.0),
    ]
    parse_mock.assert_awaited_once_with(
        model=MODEL,
        reasoning={"effort": "low"},
        instructions=RERANK_INSTRUCTIONS,
        input=build_rerank_input("Что означает C2TE S1?", candidates),
        text_format=RerankResponse,
    )


@pytest.mark.asyncio
async def test_rerank_preserves_rrf_order_for_equal_relevance_scores() -> None:
    first = make_candidate("first", score=0.0328)
    second = make_candidate("second", score=0.0320)
    parse_mock = AsyncMock(
        return_value=make_response(
            [
                RerankItem(chunk_id=second.chunk.id, relevance_score=2),
                RerankItem(chunk_id=first.chunk.id, relevance_score=2),
            ]
        )
    )
    reranker = OpenAIReranker(client=make_client(parse_mock), model=MODEL)

    results = await reranker.rerank("Вопрос", [first, second])

    assert [result.chunk.id for result in results] == ["first", "second"]


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
@pytest.mark.asyncio
async def test_rerank_rejects_blank_query_without_calling_openai(query: str) -> None:
    parse_mock = AsyncMock()
    reranker = OpenAIReranker(client=make_client(parse_mock), model=MODEL)

    with pytest.raises(ValueError, match="query must not be blank"):
        await reranker.rerank(query, [make_candidate("chunk", score=0.1)])

    parse_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_rerank_returns_empty_list_without_calling_openai() -> None:
    parse_mock = AsyncMock()
    reranker = OpenAIReranker(client=make_client(parse_mock), model=MODEL)

    results = await reranker.rerank("Вопрос", [])

    assert results == []
    parse_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_rerank_rejects_duplicate_candidate_ids_without_calling_openai() -> None:
    parse_mock = AsyncMock()
    reranker = OpenAIReranker(client=make_client(parse_mock), model=MODEL)
    candidates = [
        make_candidate("duplicate", score=0.2),
        make_candidate("duplicate", score=0.1),
    ]

    with pytest.raises(ValueError, match="candidate chunk ids must be unique"):
        await reranker.rerank("Вопрос", candidates)

    parse_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_rerank_rejects_missing_structured_output() -> None:
    parse_mock = AsyncMock(return_value=SimpleNamespace(output_parsed=None))
    reranker = OpenAIReranker(client=make_client(parse_mock), model=MODEL)

    with pytest.raises(
        LLMInvalidResponseError,
        match="did not return structured reranking output",
    ):
        await reranker.rerank(
            "Вопрос",
            [make_candidate("chunk", score=0.1)],
        )


@pytest.mark.parametrize(
    "items",
    [
        [
            RerankItem(chunk_id="first", relevance_score=3),
            RerankItem(chunk_id="first", relevance_score=2),
        ],
        [RerankItem(chunk_id="first", relevance_score=3)],
        [
            RerankItem(chunk_id="first", relevance_score=3),
            RerankItem(chunk_id="unknown", relevance_score=2),
        ],
    ],
    ids=["duplicate", "missing", "unknown"],
)
@pytest.mark.asyncio
async def test_rerank_rejects_output_ids_that_do_not_match_candidates(
    items: list[RerankItem],
) -> None:
    parse_mock = AsyncMock(return_value=make_response(items))
    reranker = OpenAIReranker(client=make_client(parse_mock), model=MODEL)
    candidates = [
        make_candidate("first", score=0.2),
        make_candidate("second", score=0.1),
    ]

    with pytest.raises(
        LLMInvalidResponseError,
        match="unexpected reranking chunk ids",
    ):
        await reranker.rerank("Вопрос", candidates)


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (APITimeoutError(REQUEST), LLMTimeoutError),
        (APIConnectionError(request=REQUEST), LLMUnavailableError),
        (
            RateLimitError(
                "Rate limit",
                response=httpx2.Response(429, request=REQUEST),
                body=None,
            ),
            LLMRateLimitError,
        ),
        (
            AuthenticationError(
                "Invalid key",
                response=httpx2.Response(401, request=REQUEST),
                body=None,
            ),
            LLMAuthenticationError,
        ),
        (APIError("Unknown OpenAI error", REQUEST, body=None), LLMUnavailableError),
    ],
)
@pytest.mark.asyncio
async def test_rerank_maps_openai_errors(
    sdk_error: APIError,
    expected_error: type[LLMError],
) -> None:
    parse_mock = AsyncMock(side_effect=sdk_error)
    reranker = OpenAIReranker(client=make_client(parse_mock), model=MODEL)

    with pytest.raises(expected_error) as captured:
        await reranker.rerank(
            "Вопрос",
            [make_candidate("chunk", score=0.1)],
        )

    assert captured.value.__cause__ is sdk_error
