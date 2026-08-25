import json
from dataclasses import dataclass
from typing import Protocol

import tiktoken

from app.rag.models import SearchResult


class TokenCounter(Protocol):
    def count_tokens(self, text: str) -> int: ...


class TiktokenTokenCounter:
    def __init__(self, encoding_name: str) -> None:
        self._encoding_name = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding_name.encode(text))


@dataclass(frozen=True, slots=True)
class RagContext:
    text: str
    sources: list[SearchResult]
    token_count: int


def serialize_context_payload(results: list[SearchResult]) -> str:
    payload = [
        {"id": result.chunk.id, "title": result.chunk.title, "text": result.chunk.text}
        for result in results
    ]

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_rag_context(
    results: list[SearchResult], *, token_counter: TokenCounter, max_tokens: int
) -> RagContext:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    selected_results: list[SearchResult] = []
    serialized_context = serialize_context_payload([])
    token_count = token_counter.count_tokens(serialized_context)

    for result in results:
        candidate_results = [*selected_results, result]
        candidate_context = serialize_context_payload(candidate_results)
        candidate_token_count = token_counter.count_tokens(candidate_context)

        if candidate_token_count > max_tokens:
            continue

        selected_results.append(result)
        serialized_context = candidate_context
        token_count = candidate_token_count

    return RagContext(
        text=serialized_context, sources=selected_results, token_count=token_count
    )
