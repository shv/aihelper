import math

import pytest

from app.rag.bm25 import Bm25Index, tokenize
from app.rag.models import DocumentChunk

K1 = 1.2
B = 0.75


def make_chunk(
    chunk_id: str,
    *,
    title: str,
    text: str,
    category: str,
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        title=title,
        text=text,
        metadata={"category": category},
    )


def test_tokenize_normalizes_case_and_preserves_technical_codes() -> None:
    assert tokenize("Клей C2TE S1, ГОСТ 12.1.019-2017!") == [
        "клей",
        "c2te",
        "s1",
        "гост",
        "12.1.019-2017",
    ]


def test_search_returns_exact_code_match_and_omits_zero_score_documents() -> None:
    exact_match = make_chunk(
        "adhesive-c2te-s1",
        title="Эластичный плиточный клей",
        text="Класс клея C2TE S1.",
        category="tile",
    )
    other_document = make_chunk(
        "waterproofing",
        title="Подготовка мокрой зоны",
        text="Основание грунтуют и гидроизолируют.",
        category="tile",
    )
    index = Bm25Index([other_document, exact_match], k1=K1, b=B)

    results = index.search("C2TE S1", top_k=3, category=None)

    assert [result.chunk for result in results] == [exact_match]
    assert results[0].score > 0


def test_search_uses_title_as_part_of_document() -> None:
    chunk = make_chunk(
        "adhesive-c2te-s1",
        title="Клей C2TE S1",
        text="Эластичный состав для плитки.",
        category="tile",
    )
    index = Bm25Index([chunk], k1=K1, b=B)

    results = index.search("C2TE", top_k=1, category=None)

    assert [result.chunk for result in results] == [chunk]


def test_score_matches_positive_okapi_bm25_formula() -> None:
    chunk = make_chunk(
        "adhesive",
        title="",
        text="клей c2te",
        category="tile",
    )
    index = Bm25Index([chunk], k1=K1, b=B)

    [result] = index.search("c2te", top_k=1, category=None)

    expected_idf = math.log(1 + (1 - 1 + 0.5) / (1 + 0.5))
    assert result.score == pytest.approx(expected_idf)


def test_search_filters_candidates_by_category() -> None:
    tile_chunk = make_chunk(
        "tile-material",
        title="Материал C2TE",
        text="Плиточный клей.",
        category="tile",
    )
    gkl_chunk = make_chunk(
        "gkl-material",
        title="Материал C2TE",
        text="Тестовое описание.",
        category="gkl",
    )
    index = Bm25Index([gkl_chunk, tile_chunk], k1=K1, b=B)

    results = index.search("C2TE", top_k=3, category="tile")

    assert [result.chunk for result in results] == [tile_chunk]


def test_search_uses_chunk_id_for_deterministic_tie_breaking() -> None:
    chunk_b = make_chunk(
        "b-chunk",
        title="Материал C2TE",
        text="Описание.",
        category="tile",
    )
    chunk_a = make_chunk(
        "a-chunk",
        title="Материал C2TE",
        text="Описание.",
        category="tile",
    )
    index = Bm25Index([chunk_b, chunk_a], k1=K1, b=B)

    results = index.search("C2TE", top_k=2, category=None)

    assert [result.chunk.id for result in results] == ["a-chunk", "b-chunk"]


def test_search_respects_top_k() -> None:
    chunks = [
        make_chunk(
            f"chunk-{index}",
            title="Материал C2TE",
            text="Описание.",
            category="tile",
        )
        for index in range(3)
    ]
    index = Bm25Index(chunks, k1=K1, b=B)

    results = index.search("C2TE", top_k=2, category=None)

    assert len(results) == 2


@pytest.mark.parametrize("k1", [-0.1, 0.0])
def test_rejects_non_positive_k1(k1: float) -> None:
    with pytest.raises(ValueError, match="k1 must be positive"):
        Bm25Index([], k1=k1, b=B)


@pytest.mark.parametrize("b", [-0.1, 1.1])
def test_rejects_b_outside_unit_interval(b: float) -> None:
    with pytest.raises(ValueError, match="b must be between 0 and 1"):
        Bm25Index([], k1=K1, b=b)


@pytest.mark.parametrize("b", [0.0, 1.0])
def test_accepts_b_at_unit_interval_boundaries(b: float) -> None:
    Bm25Index([], k1=K1, b=b)


@pytest.mark.parametrize("top_k", [-1, 0])
def test_rejects_non_positive_top_k(top_k: int) -> None:
    index = Bm25Index([], k1=K1, b=B)

    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search("клей", top_k=top_k, category=None)


@pytest.mark.parametrize("query", ["", "!!!"])
def test_empty_or_tokenless_query_returns_no_results(query: str) -> None:
    chunk = make_chunk(
        "adhesive",
        title="Плиточный клей",
        text="Класс C2TE.",
        category="tile",
    )
    index = Bm25Index([chunk], k1=K1, b=B)

    assert index.search(query, top_k=3, category=None) == []


def test_empty_index_returns_no_results() -> None:
    index = Bm25Index([], k1=K1, b=B)

    assert index.search("C2TE", top_k=3, category=None) == []
