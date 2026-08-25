import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from app.rag.context import (
    RagContext,
    TiktokenTokenCounter,
    build_rag_context,
    serialize_context_payload,
)
from app.rag.models import DocumentChunk, SearchResult


@dataclass
class CharacterTokenCounter:
    calls: list[str] = field(default_factory=list)

    def count_tokens(self, text: str) -> int:
        self.calls.append(text)
        return len(text)


def make_result(
    chunk_id: str,
    *,
    title: str,
    text: str,
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            title=title,
            text=text,
            metadata={"category": "tile", "tenant_id": "private"},
        ),
        score=3.0,
    )


def test_serialize_context_payload_includes_only_evidence_fields() -> None:
    result = make_result(
        "tile-waterproofing",
        title="Подготовка мокрой зоны",
        text="Перед плиткой выполнить гидроизоляцию.",
    )

    serialized = serialize_context_payload([result])

    assert json.loads(serialized) == [
        {
            "id": "tile-waterproofing",
            "title": "Подготовка мокрой зоны",
            "text": "Перед плиткой выполнить гидроизоляцию.",
        }
    ]
    assert "score" not in serialized
    assert "metadata" not in serialized
    assert "tenant_id" not in serialized


def test_serialize_context_payload_uses_compact_unicode_json() -> None:
    result = make_result(
        "waterproofing",
        title='Гидроизоляция "мокрой зоны"',
        text="Первый слой.\nВторой слой.",
    )

    serialized = serialize_context_payload([result])

    assert "Гидроизоляция" in serialized
    assert "\\u0413" not in serialized
    assert "}, {" not in serialized
    assert ": " not in serialized
    assert json.loads(serialized)[0]["text"] == "Первый слой.\nВторой слой."


def test_build_context_returns_empty_json_and_its_token_count() -> None:
    counter = CharacterTokenCounter()

    context = build_rag_context([], token_counter=counter, max_tokens=10)

    assert context == RagContext(text="[]", sources=[], token_count=2)
    assert counter.calls == ["[]"]


def test_build_context_includes_document_at_exact_budget_boundary() -> None:
    result = make_result("first", title="First", text="First text")
    serialized = serialize_context_payload([result])

    context = build_rag_context(
        [result],
        token_counter=CharacterTokenCounter(),
        max_tokens=len(serialized),
    )

    assert context.text == serialized
    assert context.sources == [result]
    assert context.token_count == len(serialized)


def test_build_context_excludes_document_one_token_over_budget() -> None:
    result = make_result("first", title="First", text="First text")
    serialized = serialize_context_payload([result])

    context = build_rag_context(
        [result],
        token_counter=CharacterTokenCounter(),
        max_tokens=len(serialized) - 1,
    )

    assert context == RagContext(text="[]", sources=[], token_count=2)


def test_build_context_skips_oversized_first_document_and_keeps_smaller_second() -> (
    None
):
    oversized = make_result(
        "oversized",
        title="Oversized",
        text="x" * 500,
    )
    small = make_result("small", title="Small", text="Short")
    small_context = serialize_context_payload([small])

    context = build_rag_context(
        [oversized, small],
        token_counter=CharacterTokenCounter(),
        max_tokens=len(small_context),
    )

    assert context.sources == [small]
    assert context.text == small_context
    assert "oversized" not in context.text


def test_build_context_skips_oversized_middle_document_and_preserves_order() -> None:
    first = make_result("first", title="First", text="Short first")
    oversized = make_result("oversized", title="Oversized", text="x" * 500)
    third = make_result("third", title="Third", text="Short third")
    expected_text = serialize_context_payload([first, third])

    context = build_rag_context(
        [first, oversized, third],
        token_counter=CharacterTokenCounter(),
        max_tokens=len(expected_text),
    )

    assert context.sources == [first, third]
    assert [item["id"] for item in json.loads(context.text)] == ["first", "third"]
    assert context.token_count == len(expected_text)


def test_build_context_never_truncates_selected_document() -> None:
    text = "Полное предложение с важным условием безопасности."
    result = make_result("safety", title="Безопасность", text=text)
    serialized = serialize_context_payload([result])

    context = build_rag_context(
        [result],
        token_counter=CharacterTokenCounter(),
        max_tokens=len(serialized),
    )

    assert json.loads(context.text)[0]["text"] == text


@pytest.mark.parametrize("max_tokens", [-1, 0])
def test_build_context_rejects_non_positive_budget(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        build_rag_context(
            [],
            token_counter=CharacterTokenCounter(),
            max_tokens=max_tokens,
        )


def test_tiktoken_counter_uses_selected_encoding() -> None:
    text = "Подготовка стен перед укладкой плитки"
    encoding = MagicMock()
    encoding.encode.return_value = [10, 20, 30, 40]

    with patch(
        "app.rag.context.tiktoken.get_encoding",
        return_value=encoding,
    ) as get_encoding:
        counter = TiktokenTokenCounter(encoding_name="o200k_base")
        token_count = counter.count_tokens(text)

    get_encoding.assert_called_once_with("o200k_base")
    encoding.encode.assert_called_once_with(text)
    assert token_count == 4


def test_tiktoken_counter_rejects_unknown_encoding() -> None:
    with (
        patch(
            "app.rag.context.tiktoken.get_encoding",
            side_effect=ValueError("unknown encoding"),
        ),
        pytest.raises(ValueError, match="unknown encoding"),
    ):
        TiktokenTokenCounter(encoding_name="unknown-encoding")
