import math

from app.rag.models import EmbeddedChunk, SearchResult


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        raise ValueError("Embeddings must not be empty")

    if len(left) != len(right):
        raise ValueError("Embeddings must have equal dimensions")

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        raise ValueError("Embedding norm must not be zero")

    return dot_product / (left_norm * right_norm)


def search_top_k(
    query_embedding: list[float], chunks: list[EmbeddedChunk], *, top_k: int
) -> list[SearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    results = [
        SearchResult(
            chunk=item.chunk, score=cosine_similarity(query_embedding, item.embedding)
        )
        for item in chunks
    ]

    results.sort(key=lambda result: result.score, reverse=True)

    return results[:top_k]
