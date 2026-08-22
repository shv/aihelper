import pytest

from app.rag.fusion import reciprocal_rank_fusion
from app.rag.models import DocumentChunk, SearchResult

RRF_K = 60


def make_result(chunk_id: str, *, score: float) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            title=f"Title {chunk_id}",
            text=f"Text {chunk_id}",
            metadata={"category": "test"},
        ),
        score=score,
    )


def test_fusion_boosts_document_found_by_both_rankings() -> None:
    shared = make_result("shared", score=0.01)
    vector_only = make_result("vector-only", score=100.0)
    bm25_only = make_result("bm25-only", score=0.001)

    results = reciprocal_rank_fusion(
        [
            [shared, vector_only],
            [bm25_only, shared],
        ],
        rrf_k=RRF_K,
        top_k=3,
    )

    assert [result.chunk.id for result in results] == [
        "shared",
        "bm25-only",
        "vector-only",
    ]
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert results[1].score == pytest.approx(1 / 61)
    assert results[2].score == pytest.approx(1 / 62)


def test_fusion_uses_rank_positions_instead_of_original_scores() -> None:
    first = make_result("first", score=-1_000.0)
    second = make_result("second", score=1_000.0)

    results = reciprocal_rank_fusion(
        [[first, second]],
        rrf_k=RRF_K,
        top_k=2,
    )

    assert [result.chunk.id for result in results] == ["first", "second"]
    assert results[0].score == pytest.approx(1 / 61)
    assert results[1].score == pytest.approx(1 / 62)


def test_fusion_preserves_results_found_by_only_one_retriever() -> None:
    vector_only = make_result("vector-only", score=0.9)
    bm25_only = make_result("bm25-only", score=5.0)

    results = reciprocal_rank_fusion(
        [[vector_only], [bm25_only]],
        rrf_k=RRF_K,
        top_k=2,
    )

    assert {result.chunk.id for result in results} == {
        "vector-only",
        "bm25-only",
    }


def test_duplicate_chunk_inside_one_ranking_contributes_only_once() -> None:
    duplicate = make_result("duplicate", score=0.9)

    [result] = reciprocal_rank_fusion(
        [[duplicate, duplicate]],
        rrf_k=RRF_K,
        top_k=2,
    )

    assert result.chunk.id == "duplicate"
    assert result.score == pytest.approx(1 / 61)


def test_fusion_uses_chunk_id_for_deterministic_tie_breaking() -> None:
    chunk_b = make_result("b-chunk", score=0.9)
    chunk_a = make_result("a-chunk", score=5.0)

    results = reciprocal_rank_fusion(
        [[chunk_b], [chunk_a]],
        rrf_k=RRF_K,
        top_k=2,
    )

    assert [result.chunk.id for result in results] == ["a-chunk", "b-chunk"]


def test_fusion_respects_top_k() -> None:
    results = reciprocal_rank_fusion(
        [
            [
                make_result("first", score=3.0),
                make_result("second", score=2.0),
                make_result("third", score=1.0),
            ]
        ],
        rrf_k=RRF_K,
        top_k=2,
    )

    assert [result.chunk.id for result in results] == ["first", "second"]


@pytest.mark.parametrize("rrf_k", [-1, 0])
def test_fusion_rejects_non_positive_rrf_k(rrf_k: int) -> None:
    with pytest.raises(ValueError, match="rrf_k must be positive"):
        reciprocal_rank_fusion([], rrf_k=rrf_k, top_k=3)


@pytest.mark.parametrize("top_k", [-1, 0])
def test_fusion_rejects_non_positive_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        reciprocal_rank_fusion([], rrf_k=RRF_K, top_k=top_k)


def test_fusion_returns_empty_list_for_empty_rankings() -> None:
    assert (
        reciprocal_rank_fusion(
            [[], []],
            rrf_k=RRF_K,
            top_k=3,
        )
        == []
    )
