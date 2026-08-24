import pytest
from pydantic import ValidationError

from app.llm.exceptions import LLMInvalidResponseError
from app.rag.citation import validate_citations
from app.rag.models import DocumentChunk, SearchResult
from app.schemas import GroundedRepairAdvice, RagCitation


def make_source(
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
            metadata={"category": "tile"},
        ),
        score=3.0,
    )


def test_grounded_advice_accepts_non_empty_citations() -> None:
    citation = RagCitation(
        source_id="c2te-s1",
        quote="C — цементная основа",
    )

    advice = GroundedRepairAdvice(
        summary="C означает цементную основу.",
        clarifying_questions=[],
        recommendations=[],
        risks=[],
        requires_professional=False,
        citations=[citation],
    )

    assert advice.citations == [citation]


def test_grounded_advice_rejects_empty_citations() -> None:
    with pytest.raises(ValidationError):
        GroundedRepairAdvice(
            summary="Ответ без доказательств",
            clarifying_questions=[],
            recommendations=[],
            risks=[],
            requires_professional=False,
            citations=[],
        )


@pytest.mark.parametrize("field", ["source_id", "quote"])
def test_citation_rejects_empty_required_string(field: str) -> None:
    payload = {
        "source_id": "source",
        "quote": "Цитата",
    }
    payload[field] = ""

    with pytest.raises(ValidationError):
        RagCitation.model_validate(payload)


def test_citation_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RagCitation.model_validate(
            {
                "source_id": "source",
                "quote": "Цитата",
                "url": "https://example.com",
            }
        )


def test_validate_citations_accepts_exact_quote_from_source_text() -> None:
    source = make_source(
        "c2te-s1",
        title="Расшифровка C2TE S1",
        text="C — цементная основа. T — сниженное сползание.",
    )
    citations = [
        RagCitation(
            source_id="c2te-s1",
            quote="C — цементная основа",
        )
    ]

    validate_citations(citations, [source])


def test_validate_citations_accepts_distinct_quotes_from_same_source() -> None:
    source = make_source(
        "c2te-s1",
        title="Расшифровка C2TE S1",
        text="C — цементная основа. T — сниженное сползание.",
    )
    citations = [
        RagCitation(source_id="c2te-s1", quote="C — цементная основа"),
        RagCitation(source_id="c2te-s1", quote="T — сниженное сползание"),
    ]

    validate_citations(citations, [source])


def test_validate_citations_rejects_unknown_source_id() -> None:
    source = make_source(
        "known",
        title="Известный источник",
        text="Проверенный текст.",
    )
    citation = RagCitation(source_id="invented", quote="Проверенный текст")

    with pytest.raises(
        LLMInvalidResponseError,
        match="Unknown citation source id: invented",
    ) as captured:
        validate_citations([citation], [source])

    assert isinstance(captured.value.__cause__, KeyError)


def test_validate_citations_rejects_quote_not_present_verbatim() -> None:
    source = make_source(
        "c2te-s1",
        title="Расшифровка C2TE S1",
        text="C — цементная основа.",
    )
    citation = RagCitation(
        source_id="c2te-s1",
        quote="C означает цементную основу.",
    )

    with pytest.raises(
        LLMInvalidResponseError,
        match="Citation is not an exact quote from source: c2te-s1",
    ):
        validate_citations([citation], [source])


def test_validate_citations_does_not_accept_quote_found_only_in_title() -> None:
    source = make_source(
        "c2te-s1",
        title="Расшифровка C2TE S1",
        text="C — цементная основа.",
    )
    citation = RagCitation(
        source_id="c2te-s1",
        quote="Расшифровка C2TE S1",
    )

    with pytest.raises(LLMInvalidResponseError):
        validate_citations([citation], [source])


def test_validate_citations_is_case_sensitive() -> None:
    source = make_source(
        "c2te-s1",
        title="Расшифровка",
        text="Cementitious означает цементную основу.",
    )
    citation = RagCitation(
        source_id="c2te-s1",
        quote="cementitious означает цементную основу",
    )

    with pytest.raises(LLMInvalidResponseError):
        validate_citations([citation], [source])


def test_validate_citations_rejects_duplicate_citation() -> None:
    source = make_source(
        "c2te-s1",
        title="Расшифровка",
        text="C — цементная основа.",
    )
    citation = RagCitation(
        source_id="c2te-s1",
        quote="C — цементная основа",
    )

    with pytest.raises(
        LLMInvalidResponseError,
        match="Duplicate citation for source id: c2te-s1",
    ):
        validate_citations([citation, citation], [source])


def test_validate_citations_allows_same_quote_from_different_sources() -> None:
    shared_text = "Основание необходимо загрунтовать."
    first_source = make_source(
        "first",
        title="Первый источник",
        text=shared_text,
    )
    second_source = make_source(
        "second",
        title="Второй источник",
        text=shared_text,
    )
    citations = [
        RagCitation(source_id="first", quote=shared_text),
        RagCitation(source_id="second", quote=shared_text),
    ]

    validate_citations(citations, [first_source, second_source])


def test_validate_citations_rejects_duplicate_source_ids() -> None:
    first_source = make_source(
        "duplicate",
        title="Первый источник",
        text="Первый текст.",
    )
    second_source = make_source(
        "duplicate",
        title="Второй источник",
        text="Второй текст.",
    )

    with pytest.raises(ValueError, match="source chunk ids must be unique"):
        validate_citations([], [first_source, second_source])


def test_validate_citations_accepts_empty_citations_and_sources() -> None:
    validate_citations([], [])
