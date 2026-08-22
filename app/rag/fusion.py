from app.rag.models import DocumentChunk, SearchResult


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]], *, rrf_k: int, top_k: int
) -> list[SearchResult]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    scores: dict[str, float] = {}
    chunks: dict[str, DocumentChunk] = {}

    for ranking in rankings:
        seen_chunk_ids: set[str] = set()

        for rank, result in enumerate(ranking, start=1):
            chunk_id = result.chunk.id

            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)
            chunks.setdefault(chunk_id, result.chunk)

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + rank)

    fused_results = [
        SearchResult(chunk=chunks[chunk_id], score=score)
        for chunk_id, score in scores.items()
    ]

    fused_results.sort(key=lambda result: (-result.score, result.chunk.id))

    return fused_results[:top_k]
