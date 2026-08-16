import hashlib
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

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.embedding_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: DocumentChunk
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True, slots=True)
class IndexedChunkState:
    content_hash: str
    embedding_model: str
    metadata: dict[str, str]
