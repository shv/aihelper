import pytest

from app.rag.models import DocumentChunk, EmbeddedChunk
from app.rag.search import cosine_similarity, search_top_k


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [2.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
    ],
)
def test_cosine_similarity(
    left: list[float],
    right: list[float],
    expected: float,
) -> None:
    assert cosine_similarity(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([], []),
        ([1.0], [1.0, 2.0]),
        ([0.0, 0.0], [1.0, 0.0]),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(
    left: list[float],
    right: list[float],
) -> None:
    with pytest.raises(ValueError):
        cosine_similarity(left, right)


def make_chunk(
    chunk_id: str,
    embedding: list[float],
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=DocumentChunk(
            id=chunk_id,
            title=chunk_id,
            text=chunk_id,
        ),
        embedding=embedding,
    )


def test_search_top_k_orders_results_by_similarity() -> None:
    chunks = [
        make_chunk("orthogonal", [0.0, 1.0]),
        make_chunk("closest", [1.0, 0.0]),
        make_chunk("second", [0.8, 0.6]),
    ]

    results = search_top_k(
        query_embedding=[1.0, 0.0],
        chunks=chunks,
        top_k=2,
    )

    assert [result.chunk.id for result in results] == [
        "closest",
        "second",
    ]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.8)


def test_search_top_k_rejects_non_positive_value() -> None:
    with pytest.raises(ValueError):
        search_top_k(
            query_embedding=[1.0, 0.0],
            chunks=[],
            top_k=0,
        )
