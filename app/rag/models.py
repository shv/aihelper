from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: DocumentChunk
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: DocumentChunk
    score: float
