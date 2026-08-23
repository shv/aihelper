import json
from typing import Protocol

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field

from app.llm.exceptions import (
    LLMAuthenticationError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.rag.models import SearchResult


class RerankItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    relevance_score: int = Field(ge=0, le=3)


class RerankResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RerankItem]


class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: list[SearchResult]
    ) -> list[SearchResult]: ...


RERANK_INSTRUCTIONS = """
Ты переранжируешь фрагменты базы знаний относительно вопроса пользователя.

Оцени каждый фрагмент только по содержащемуся в нём тексту:

3 — фрагмент непосредственно содержит ответ на вопрос;
2 — фрагмент содержит существенную часть ответа;
1 — фрагмент относится к теме, но не отвечает на вопрос;
0 — фрагмент не относится к вопросу или не помогает ответить.

Правила:
- не используй внешние знания;
- верни каждый переданный chunk_id ровно один раз;
- не добавляй новые chunk_id;
- relevance_score должен быть целым числом от 0 до 3.
""".strip()


def build_rerank_input(query: str, candidates: list[SearchResult]) -> str:
    return json.dumps(
        {
            "query": query,
            "documents": [
                {
                    "chunk_id": candidate.chunk.id,
                    "title": candidate.chunk.title,
                    "text": candidate.chunk.text,
                }
                for candidate in candidates
            ],
        },
        ensure_ascii=False,
    )


class OpenAIReranker:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be blank")

        if not candidates:
            return []

        candidate_by_id = {candidate.chunk.id: candidate for candidate in candidates}

        if len(candidate_by_id) != len(candidates):
            raise ValueError("candidate chunk ids must be unique")

        try:
            response = await self._client.responses.parse(
                model=self._model,
                reasoning={"effort": "low"},
                instructions=RERANK_INSTRUCTIONS,
                input=build_rerank_input(query, candidates),
                text_format=RerankResponse,
            )
        except APITimeoutError as error:
            raise LLMTimeoutError("OpenAI reranking request timed out") from error
        except RateLimitError as error:
            raise LLMRateLimitError("OpenAI reranking rate limit exceeded") from error
        except AuthenticationError as error:
            raise LLMAuthenticationError(
                "OpenAI reranking authentication failed"
            ) from error
        except APIConnectionError as error:
            raise LLMUnavailableError(
                "Cannot connect to OpenAI for reranking"
            ) from error
        except APIError as error:
            raise LLMUnavailableError("OpenAI reranking API error") from error

        if response.output_parsed is None:
            raise LLMInvalidResponseError(
                "OpenAI did not return structured reranking output"
            )

        items = response.output_parsed.items
        output_ids = [item.chunk_id for item in items]

        if len(output_ids) != len(set(output_ids)):
            raise LLMInvalidResponseError(
                "OpenAI returned unexpected reranking chunk ids"
            )

        if set(output_ids) != set(candidate_by_id):
            raise LLMInvalidResponseError(
                "OpenAI returned unexpected reranking chunk ids"
            )

        original_rank = {
            candidate.chunk.id: rank for rank, candidate in enumerate(candidates)
        }

        sorted_items = sorted(
            items,
            key=lambda item: (-item.relevance_score, original_rank[item.chunk_id]),
        )

        return [
            SearchResult(
                chunk=candidate_by_id[item.chunk_id].chunk,
                score=float(item.relevance_score),
            )
            for item in sorted_items
        ]
