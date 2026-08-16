from dataclasses import dataclass
from typing import Protocol

from app.rag.embeddings import EmbeddingProvider
from app.rag.models import DocumentChunk, EmbeddedChunk, IndexedChunkState


@dataclass(frozen=True, slots=True)
class IndexingStats:
    total: int
    indexed: int
    skipped: int


class IndexStore(Protocol):
    async def get_index_state(
        self, chunk_ids: list[str]
    ) -> dict[str, IndexedChunkState]: ...

    async def upsert(
        self, chunks: list[EmbeddedChunk], *, embedding_model: str
    ) -> None: ...


class RagIndexer:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        store: IndexStore,
        *,
        embedding_model: str,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._store = store
        self._embedding_model = embedding_model

    async def index(
        self,
        chunks: list[DocumentChunk],
    ) -> IndexingStats:
        index_state = await self._store.get_index_state([chunk.id for chunk in chunks])
        chunks_for_index = []
        for chunk in chunks:
            state = index_state.get(chunk.id)

            if (
                state is None
                or state.content_hash != chunk.content_hash
                or state.embedding_model != self._embedding_model
                or state.metadata != chunk.metadata
            ):
                chunks_for_index.append(chunk)

        if chunks_for_index:
            chunk_embeddings = await self._embedding_provider.embed(
                [chunk.embedding_text for chunk in chunks_for_index]
            )

            embedded_chunks = [
                EmbeddedChunk(chunk=chunk, embedding=embedding)
                for chunk, embedding in zip(
                    chunks_for_index,
                    chunk_embeddings,
                    strict=True,
                )
            ]

            await self._store.upsert(
                embedded_chunks,
                embedding_model=self._embedding_model,
            )
        return IndexingStats(
            total=len(chunks),
            indexed=len(chunks_for_index),
            skipped=len(chunks) - len(chunks_for_index),
        )
