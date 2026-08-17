import json

from app.rag.context import build_rag_context
from app.rag.models import DocumentChunk, SearchResult


def make_search_result(
    chunk_id: str,
    *,
    title: str,
    text: str,
    score: float,
    metadata: dict[str, str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            title=title,
            text=text,
            metadata=metadata or {},
        ),
        score=score,
    )


def test_build_rag_context_returns_empty_json_array_for_no_results() -> None:
    context = build_rag_context([])

    assert json.loads(context) == []


def test_build_rag_context_preserves_retrieval_order() -> None:
    results = [
        make_search_result(
            "tile-waterproofing",
            title="Подготовка мокрой зоны",
            text="Перед плиткой выполнить гидроизоляцию.",
            score=0.91,
        ),
        make_search_result(
            "gkl-cabinet",
            title="Крепление шкафа к ГКЛ",
            text="Крепить шкаф к стойкам или закладной.",
            score=0.76,
        ),
    ]

    context = json.loads(build_rag_context(results))

    assert [item["id"] for item in context] == [
        "tile-waterproofing",
        "gkl-cabinet",
    ]


def test_build_rag_context_includes_only_llm_evidence_fields() -> None:
    result = make_search_result(
        "tile-waterproofing",
        title="Подготовка мокрой зоны",
        text="Перед плиткой выполнить гидроизоляцию.",
        score=0.91,
        metadata={"category": "tile", "tenant_id": "private-tenant"},
    )

    context = json.loads(build_rag_context([result]))

    assert context == [
        {
            "id": "tile-waterproofing",
            "title": "Подготовка мокрой зоны",
            "text": "Перед плиткой выполнить гидроизоляцию.",
        }
    ]
    assert set(context[0]) == {"id", "title", "text"}


def test_build_rag_context_produces_valid_json_without_escaping_cyrillic() -> None:
    title = 'Гидроизоляция "мокрой зоны"'
    text = "Первый слой.\nВторой слой."
    result = make_search_result(
        "waterproofing",
        title=title,
        text=text,
        score=1.0,
    )

    serialized_context = build_rag_context([result])
    parsed_context = json.loads(serialized_context)

    assert "Гидроизоляция" in serialized_context
    assert "\\u0413" not in serialized_context
    assert parsed_context[0]["title"] == title
    assert parsed_context[0]["text"] == text
