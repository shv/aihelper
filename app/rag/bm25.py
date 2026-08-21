import math
import re
from collections import Counter

from .models import DocumentChunk, SearchResult

TOKEN_PATTERN = re.compile(r"[0-9a-zа-яё]+(?:[.-][0-9a-zа-яё]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class Bm25Index:
    def __init__(self, chunks: list[DocumentChunk], *, k1: float, b: float) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self._chunks = chunks
        self._k1 = k1
        self._b = b

        document_tokens = [tokenize(chunk.embedding_text) for chunk in chunks]

        self._term_frequencies = [Counter(tokens) for tokens in document_tokens]

        self._document_lengths = [len(tokens) for tokens in document_tokens]

        self._document_frequencies: Counter[str] = Counter()

        for term_frequency in self._term_frequencies:
            self._document_frequencies.update(term_frequency.keys())

        document_count = len(chunks)

        self._average_document_length = (
            sum(self._document_lengths) / document_count if document_count > 0 else 0.0
        )

    def search(
        self, query: str, *, top_k: int, category: str | None = None
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_terms = set(tokenize(query))

        if not query_terms:
            return []

        results: list[SearchResult] = []

        for document_index, chunk in enumerate(self._chunks):
            if category is not None and chunk.metadata.get("category") != category:
                continue

            score = self._score_document(document_index, query_terms)

            if score <= 0:
                continue

            results.append(SearchResult(chunk=chunk, score=score))

        results.sort(key=lambda result: (-result.score, result.chunk.id))

        return results[:top_k]

    def _score_document(self, document_index: int, query_terms: set[str]) -> float:
        document_count = len(self._chunks)

        if document_count == 0:
            return 0.0

        term_frequencies = self._term_frequencies[document_index]
        document_length = self._document_lengths[document_index]

        score = 0.0

        for term in query_terms:
            term_frequency = term_frequencies.get(term, 0)

            if term_frequency == 0:
                continue

            document_frequency = self._document_frequencies[term]

            inverse_document_frequency = math.log(
                1
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

            length_normalization = (
                1 - self._b + self._b * document_length / self._average_document_length
            )

            term_score = (
                inverse_document_frequency
                * term_frequency
                * (self._k1 + 1)
                / (term_frequency + self._k1 * length_normalization)
            )

            score += term_score

        return score
