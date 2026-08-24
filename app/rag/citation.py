from app.llm.exceptions import LLMInvalidResponseError
from app.rag.models import SearchResult
from app.schemas import RagCitation


def validate_citations(
    citations: list[RagCitation], sources: list[SearchResult]
) -> None:
    source_by_id = {source.chunk.id: source for source in sources}

    if len(source_by_id) != len(sources):
        raise ValueError("source chunk ids must be unique")

    seen_citation: set[tuple[str, str]] = set()

    for citation in citations:
        try:
            source = source_by_id[citation.source_id]
        except KeyError as error:
            raise LLMInvalidResponseError(
                f"Unknown citation source id: {citation.source_id}"
            ) from error

        citation_key = (citation.source_id, citation.quote)

        if citation_key in seen_citation:
            raise LLMInvalidResponseError(
                f"Duplicate citation for source id: {citation.source_id}"
            )

        seen_citation.add(citation_key)

        if citation.quote not in source.chunk.text:
            raise LLMInvalidResponseError(
                f"Citation is not an exact quote from source: {citation.source_id}"
            )
